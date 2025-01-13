from contextlib import ExitStack
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
import re
from subprocess import PIPE, Popen, TimeoutExpired
import sys
from time import time

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.storage import storages
from django.db import models
from django.db.transaction import atomic

from mibios.data import TableConfig
from mibios.ncbi_taxonomy.models import TaxNode
from mibios.umrad.fields import AccessionField
from mibios.umrad.model_utils import (
    digits, opt, ch_opt, fk_req, fk_opt, uniq_opt, Model,
)
from mibios.umrad.models import CompoundRecord, FuncRefDBEntry, UniRef100
from mibios.umrad.manager import Manager
from mibios.umrad.utils import ProgressPrinter

from . import managers, get_sample_model, sra
from .amplicon import get_target_genes, quick_analysis, quick_annotation
from .fields import DataPathField, ReadOnlyFileField
from .queryset import FileQuerySet, SampleQuerySet
from .utils import get_fasta_sequence


log = getLogger(__name__)


class AbstractAbundance(Model):
    """
    abundance vs <something>

    With data from the Sample_xxxx_<something>_VERSION.txt files
    """
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    scos = models.DecimalField(**digits(12, 2))
    rpkm = models.DecimalField(**digits(12, 2))
    # lca ?

    class Meta(Model.Meta):
        abstract = True


class AbstractSample(Model):
    TYPE_AMPLICON = 'amplicon'
    TYPE_METAGENOME = 'metagenome'
    TYPE_METATRANS = 'metatranscriptome'
    TYPE_ISOLATE = 'isolate_genome'
    SAMPLE_TYPES_CHOICES = (
        (TYPE_AMPLICON, TYPE_AMPLICON),
        (TYPE_METAGENOME, TYPE_METAGENOME),
        (TYPE_METATRANS, TYPE_METATRANS),
        (TYPE_ISOLATE, TYPE_ISOLATE),
    )

    sample_id = models.CharField(
        max_length=32,
        unique=True,
        help_text='internal sample accession',
    )
    sample_name = models.TextField(
        max_length=32,
        **ch_opt,
        help_text='sample ID or name as given by original data source',
    )
    dataset = models.ForeignKey(
        settings.OMICS_DATASET_MODEL,
        **fk_req,
    )
    sample_type = models.CharField(
        max_length=32,
        choices=SAMPLE_TYPES_CHOICES,
        **opt,
    )
    has_paired_data = models.BooleanField(**opt)
    sra_accession = models.TextField(max_length=16, **ch_opt)
    amplicon_target = models.TextField(max_length=16, **ch_opt)
    fwd_primer = models.TextField(max_length=32, **ch_opt)
    rev_primer = models.TextField(max_length=32, **ch_opt)

    analysis_dir = models.TextField(
        **opt,
        help_text='path to results of analysis, relative to '
        'OMICS_PIPELINE_DATA',
    )
    # mapping data / header items from bbmap output:
    read_count = models.PositiveIntegerField(
        **opt,
        help_text='number of read pairs (post-QC) used for assembly mapping',
    )
    reads_mapped_contigs = models.PositiveIntegerField(
        **opt,
        help_text='number of reads mapped to contigs',
    )
    reads_mapped_genes = models.PositiveIntegerField(
        **opt,
        help_text='number of reads mapped to genes',
    )

    objects = Manager.from_queryset(SampleQuerySet)()
    loader = managers.SampleLoader.from_queryset(SampleQuerySet)()

    class Meta:
        abstract = True

    def __str__(self):
        return self.sample_id

    default_internal_fields = ['id', 'analysis_dir', 'sample_id']
    """ see also the overriding get_internal_fields() """

    @classmethod
    def get_internal_fields(cls):
        """
        Return list of fields with non-public usage
        """
        return cls.default_internal_fields + [
            i.name
            for i in cls._meta.get_fields()
            if i.name.endswith(('_ok', '_loaded'))
        ]

    def get_metagenome_path(self):
        """
        Get path to data analysis / pipeline results

        DEPRECATED - use get_omics_file()
        """
        path = settings.OMICS_PIPELINE_DATA / 'metagenomes' \
            / self.sample_id
        now = time()
        for i in path.iterdir():
            if now - i.stat().st_mtime < 86400:
                # interpret as "sample is still being processed by
                # pipeline" FIXME TODO this is not a proper design
                raise RuntimeError('too new')
        return path

    def get_omics_file(self, filetype):
        """
        Convenience method to get an omics file instance

        filetype: File.Type enum object or its name
        """
        return File.objects.get_instance(self, filetype)

    def get_fq_paths(self):
        # DEPRECATED
        base = settings.OMICS_DATA_ROOT / 'READS'
        fname = f'{self.accession}_{{infix}}.fastq.gz'
        return {
            infix: base / fname.format(infix=infix)
            for infix in ['dd_fwd', 'dd_rev', 'ddtrnhnp_fwd', 'ddtrnhnp_rev']
        }

    def get_checkm_stats_path(self):
        # DEPRECATED
        return (settings.OMICS_DATA_ROOT / 'BINS' / 'CHECKM'
                / f'{self.accession}_CHECKM' / 'storage'
                / 'bin_stats.analyze.tsv')

    def get_fastq_prefix(self):
        """
        Prefix for fastq filenames for this sample
        """
        if self.sample_id is None:
            part1 = f'pk{self.pk}'
        else:
            part1 = self.sample_id

        # upstream-given name w/o funny stuff
        part2 = re.sub(r'\W+', '', self.sample_name)
        return f'{part1}-{part2}'

    def get_fastq_base(self):
        """ Get directory plus base name of fastq files """
        base = self.dataset.get_fastq_path(self.sample_type)
        return base / self.get_fastq_prefix()

    def download_fastq(self, exist_ok=False, verbose=False):
        """
        download fastq files from SRA

        Returns a triple od SRA run accession, platform, list of files.  The
        number of files in the list indicates if the data is single-end or
        paired-end.  There will be either two files or one file respectively.
        """
        fastq_base = self.get_fastq_base()
        run, platform, is_paired_end = self.get_sra_run_info()

        # if output files exist, then fastq-dump will give a cryptic error
        # message, so dealing with already existing files here:
        files = list(fastq_base.parent.glob(f'{fastq_base.name}*'))
        if files:
            if exist_ok:
                if verbose:
                    print('Some destination file(s) exist already:')
                    for i in files:
                        print(f'exists: {i}')
                if is_paired_end and len(files) >= 2:
                    # TODO: check existing file via MD5?
                    pass
                elif not is_paired_end and len(files) >= 1:
                    # TODO: check existing file via MD5?
                    pass
                else:
                    # but some file missing, trigger full download
                    files = []
            else:
                files = "\n".join([str(i) for i in files])
                raise RuntimeError(f'file(s) exist:\n{files}')

        if not files:
            files = sra.download_fastq(
                run['accession'],
                dest=fastq_base,
                exist_ok=exist_ok,
                verbose=verbose,
            )

        # Check that file names follow SRA fasterq-dump output file naming
        # conventions as expected
        if len(files) == 1:
            if self.has_paired_data:
                print(f'WARNING: {self}: {self.has_paired_data=} / expected '
                      f'two files, got: {files}')
            if files[0].name != self.get_fastq_prefix() + '.fastq':
                raise RuntimeError(f'unexpected single-end filename: {files}')
        elif len(files) == 2:
            if not self.has_paired_data and self.has_paired_data is not None:
                print(f'WARNING: {self}: {self.has_paired_data=} / expected '
                      f'single file, got: {files}')
            fnames = sorted([i.name for i in files])
            if fnames[0] != self.get_fastq_prefix() + '_1.fastq':
                raise RuntimeError(f'unexpected paired filename: {fnames[0]}')
            if fnames[1] != self.get_fastq_prefix() + '_2.fastq':
                raise RuntimeError(f'unexpected paired filename: {fnames[0]}')
        elif len(files) == 3:
            # TODO: we should expect paired data with some single reads
            raise NotImplementedError
        else:
            raise RuntimeError(
                f'unexpected number of files downloaded: {files}'
            )

        # return values are suitable for Dataset.download_fastq()
        return run['accession'], platform, is_paired_end, files

    def get_sra_run_info(self):
        if self.sra_accession.startswith(('SRR', 'SRX')):
            other = self.sra_accession
        else:
            # might be ambiguous, hope for the best
            other = None
        return sra.get_run(self.biosample, other=other)

    def amplicon_test(self, dry_run=False):
        """
        Run quick amplicon/primer location test
        """
        if self.sample_type != self.TYPE_AMPLICON:
            raise RuntimeError(
                f'method is only for {self.TYPE_AMPLICON} samples'
            )

        for i in get_target_genes():
            if i in self.amplicon_target:
                gene = i
                break
        else:
            raise RuntimeError('gene target not supported')

        data = quick_analysis(
            self.get_fastq_base().parent,
            glob=self.get_fastq_prefix() + '*.fastq',
            gene=gene,
        )
        annot = quick_annotation(data, gene)
        return annot

    def assign_analysis_unit(self, create=True):
        """
        Assign sample to analysis unit, create on if needed

        Returns the AmpliconAnalysisUnit object
        """
        '''
        obj = AmpliconAnalysisUnit.objects.get_or_create(
            dataset=self.dataset,
            # TODO
        )
        # TODO: need better localization data to do this
        '''

    def get_ready_jobs(self):
        """ Convenience method to get jobs with status READY """
        return self.__class__.loader.filter(pk=self.pk).get_ready()

    def load_omics_data(self):
        """ Convenience methodto load the sample's omics data """
        return self.__class__.loader.load_omics_data(samples=[self])


class AbstractDataset(Model):
    """
    Abstract base model for a study or similar collections of samples

    To be used by apps that implement meta data models.
    """
    dataset_id = models.PositiveIntegerField(
        unique=True,
        help_text='internal accession to data set/study/project',
    )
    short_name = models.TextField(
        max_length=128,
        **uniq_opt,
        help_text='a short name or description, for internal use, not '
                  '(necessarily) for public display',
    )

    class Meta:
        abstract = True

    def __init__(self, *args, orphan_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_group = orphan_group

    _orphan_group_obj = None
    orphan_group_description = 'ungrouped samples'
    default_internal_fields = ['id', 'dataset_id', 'short_name']

    def get_set_no(self):
        """ Get dataset number from dataset_id """
        return int(self.dataset_id.removeprefix('set_'))

    def samples(self):
        """
        Return samples in a way that also works for the orphan set
        """
        if self.orphan_group:
            return get_sample_model().objects.filter(dataset=None)
        else:
            return self.sample_set.all()

    def get_samples_url(self):
        """
        Returns mibios table interface URL for table of the study's samples
        """
        conf = TableConfig(get_sample_model())
        if self.orphan_group:
            conf.filter['dataset'] = None
        else:
            conf.filter['dataset__pk'] = self.pk
        return conf.url()

    @classmethod
    @property
    def orphans(cls):
        """
        Return the fake group of samples without a study

        The returned instance is for display purpose only, should not be saved.
        Implementing classes may want to set any useful attributes such as the
        group's name or description.
        """
        if cls._orphan_group_obj is None:
            cls._orphan_group_obj = cls(orphan_group=True)
        return cls._orphan_group_obj

    def download_fastq(self, exist_ok=False):
        """ get fastq data from SRA """
        # TODO: implement correct behaviour if exist_ok and file actually exist
        manifest = []
        for i in self.sample_set.all():
            run, platform, is_paired_end, files = i.download_fastq(exist_ok=exist_ok)  # noqa: E501
            files = [i.name for i in files]
            if len(files) == 1:
                # single reads
                read1 = files[0]
                read2 = ''
            elif len(files) == 2:
                # paired-end reads, assume file names differ by infix:
                # _1 <=> _2 following SRA fasterq-dump conventions,
                # this sorts correctly
                read1, read2 = sorted(files)
            else:
                raise ValueError('can only handle one or two files per sample')

            manifest.append((
                i.sample_id,
                i.sample_name,
                run,
                platform,
                'paired' if read2 else 'single',
                i.amplicon_target,
                read1,
                read2,
            ))

        mfile = self.get_fastq_path() / 'fastq_manifest.csv'
        with open(mfile, 'w') as out:
            for row in manifest:
                out.write('\t'.join(row))
                out.write('\n')
        print(f'manifest written to: {mfile}')

    def get_sample_type(self):
        """
        Determine sample type
        """
        stypes = set(self.sample_set.values_list('sample_type', flat=True))
        num = len(stypes)
        if num == 0:
            raise RuntimeError('Data set has no samples')
        elif num > 1:
            # TODO: do we need to support this?
            raise RuntimeError('Multiple types: {stypes}')
        return stypes.pop()

    def get_fastq_path(self, sample_type=None):
        """
        Get path to fastq data storage
        """
        if sample_type is None:
            sample_type = self.get_sample_type()

        if sample_type == AbstractSample.TYPE_AMPLICON:
            base = Path(settings.AMPLICON_PIPELINE_BASE)
        else:
            raise NotImplementedError

        # FIXME: use study_id, but that's currently GLAMR-specific ?
        return base / str(self.dataset_id)

    def prepare_amplicon_analysis(self):
        """
        Ensure that amplicon analysis can be run on dataset

        1. ensure fastq data is downloaded for all samples
        2. ensure every sample is in one analysis unit
        3. ensure we have info from preliminary analysis
        """
        ...


class ReadAbundance(Model):
    """
    Read-mapping based abundance w.r.t. UniRef100

    For data from mmseqs2's {contig_}tophit_report files
    """
    # cf. mmseqs2 easy-taxonomy output (tophit_report)
    sample = models.ForeignKey(
        settings.OMICS_SAMPLE_MODEL,
        related_name='functional_abundance',
        **fk_req,
    )
    ref = models.ForeignKey(UniRef100, **fk_req, related_name='abundance')
    unique_cov = models.DecimalField(
        **digits(4, 3),
        help_text='unique coverage of target uniqueAlignedResidues / '
        'targetLength',
    )
    target_cov = models.DecimalField(
        **digits(10, 3),
        help_text='target coverage alignedResidues / targetLength',
    )
    avg_ident = models.DecimalField(**digits(4, 3))
    read_count = models.PositiveIntegerField(
        help_text='number of sequences aligning to target',
    )
    tpm = models.FloatField(null=True, verbose_name='TPM')
    rpkm = models.FloatField(null=True, verbose_name='RPKM')

    loader = managers.ReadAbundanceLoader()

    class Meta(Model.Meta):
        unique_together = (('sample', 'ref'),)
        verbose_name = 'functional abundance'


'''
class AmpliconAnalysisUnit(Model):
    """ a collection of amplicon samples to be analysed together """
    dataset = models.ForeignKey(settings.OMICS_DATASET_MODEL, **fk_req)
    seq_platform = models.CharField(max_length=32)
    is_paired_end = models.BooleanField()
    target_gene = models.CharField(max_length=16)
    fwd_primer = models.CharField(max_length=16)
    rev_primer = models.CharField(max_length=16)
    trim_params = models.CharField(max_length=128)

    class Meta:
        unique_together = (
            ('dataset', 'seq_platform', 'is_paired_end', 'fwd_primer',
             'rev_primer',),
        )
'''


class Bin(Model):
    """ Metagenomic assembly bin """
    history = None
    name = models.TextField(max_length=20, unique=True)
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    contigs = models.ManyToManyField('Contig', related_name='bins')

    # GTDB classification
    taxon = models.TextField(max_length=100)

    # CheckM
    completeness = models.DecimalField(
        max_digits=5, decimal_places=2,
    )
    contamination = models.DecimalField(
        max_digits=5, decimal_places=2,
    )
    heterogeneity = models.DecimalField(
        max_digits=5, decimal_places=2,
        verbose_name='strain heterogeneity',
    )

    # abundance
    percent_abund = models.FloatField()
    mean_depth = models.FloatField()
    trimmed_mead_depth = models.FloatField()
    covered_bases = models.IntegerField()
    variance = models.FloatField()
    length = models.IntegerField()
    read_count = models.IntegerField()
    reads_per_base = models.FloatField()
    rpkm = models.FloatField()
    tpm = models.FloatField()

    loader = managers.BinLoader()


class File(Model):
    """ An omics product file, analysis pipeline result """

    class Type(models.IntegerChoices):
        METAG_ASM = 1, 'metagenomic assembly, fasta format'
        """ e.g. samp_14/assembly/megahit_noNORM/final.contigs.renamed.fa """
        METAT_ASM = 2, 'metatranscriptome assembly, fasta format'
        """ TODO """
        FUNC_ABUND = 3, 'functional abundance, csv format'
        """ e.g. samp_14/samp_14_tophit_report """
        TAX_ABUND = 4, 'taxonomic abundance, csv format'
        """ *_lca_abund_summarized.tsv """
        FUNC_ABUND_TPM = 5, 'functional abundance (TPM) [csv]'
        CONT_ABUND = 6, 'contig abundance [csv]'
        CONT_LCA = 7, 'contig taxonomy [csv]'
        BIN_COV = 8, 'bin abundance'
        """ {sample}/bins/coverage_drep_bins.tsv """
        BIN_CLASS_ARC = 9, 'GTDB Archaea bin classification'
        BIN_CLASS_BAC = 10, 'GTDB Bacteria bin classification'
        BIN_CHECKM = 11, 'CheckM results'
        BIN_CONTIG = 12, 'binning results'

    file_pipeline = ReadOnlyFileField(upload_to=None,
                                      storage=storages['omics_pipeline'],
                                      **ch_opt)
    file_local = models.FileField(upload_to=None,
                                  storage=storages['local_public'],
                                  **ch_opt)
    file_globus = models.FileField(upload_to=None,
                                   storage=storages['globus_public'],
                                   **ch_opt)
    filetype = models.PositiveSmallIntegerField(
        choices=Type.choices,
        verbose_name='file type',
    )
    size = models.PositiveBigIntegerField()
    md5sum = models.CharField(
        max_length=32, blank=True,
        verbose_name='MD5 sum',
    )
    modtime = models.DateTimeField(verbose_name='modification time')
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)

    objects = managers.FileManager.from_queryset(FileQuerySet)()

    pipeline_checkout = None
    """ class variable, maps relative-to-common-root paths to mtime, usually
    set via the manager """

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['sample', 'filetype'],
                name='%(app_label)s_%(class)s_sample_ftype_unique',
            ),
            models.UniqueConstraint(
                fields=['file_pipeline'],
                condition=~models.Q(file_pipeline=''),  # blank is okay
                name='%(app_label)s_%(class)s_file_pipline_unique',
            ),
            models.UniqueConstraint(
                fields=['file_local'],
                condition=~models.Q(file_local=''),  # blank is okay
                name='%(app_label)s_%(class)s_file_local_unique',
            ),
            models.UniqueConstraint(
                fields=['file_globus'],
                condition=~models.Q(file_globus=''),  # blank is okay
                name='%(app_label)s_%(class)s_file_globus_unique',
            ),
        ]

    PATH_TAILS = {
        Type.METAG_ASM:
            Path('assembly', 'megahit_noNORM', 'final.contigs.renamed.fa'),
        Type.TAX_ABUND: '{sample.sample_id}_lca_abund_summarized.tsv',
        Type.FUNC_ABUND: '{sample.sample_id}_tophit_report',
        Type.FUNC_ABUND_TPM: '{sample.sample_id}_tophit_TPM.tsv',
        Type.CONT_ABUND: '{sample.sample_id}_contig_abund.tsv',
        Type.CONT_LCA: '{sample.sample_id}_contig_lca.tsv',
        Type.BIN_COV: 'bins/coverage_drep_bins.tsv',
        Type.BIN_CLASS_ARC: 'bins/GTDB/gtdbtk.ar53.summary.tsv',
        Type.BIN_CLASS_BAC: 'bins/GTDB/gtdbtk.bac120.summary.tsv',
        Type.BIN_CHECKM: 'bins/all_raw_bins/checkm.txt',
        Type.BIN_CONTIG: 'bins/contig_bins.rds',
    }
    """ file location, relative to a sample's data dir """

    def __str__(self):
        return str(self.file_pipeline)

    def check_stat(self):
        """ Convenience method to check all fiel fields """
        for i in ('file_pipeline', 'file_local', 'file_globus'):
            self.check_stat_field(i)

    def check_stat_field(self, field_name):
        """
        Check field files' existence and size and modtime.

        Checks that the field file exists in storage. Validates the size and
        modtime fields against the store file.  For file_pipeline, if size
        and/or modtime are not set yet, then those fields will be auto-filled
        instead.

        Raise ValidationError if the file field is blank or if the stat call on
        the file fails (usually if the file does not exist in storage.
        """
        field_file = getattr(self, field_name)
        if not field_file:
            if field_name == 'file_pipeline':
                raise ValidationError({field_name: 'field is blank'})
            return
        try:
            stat = Path(field_file.path).stat()
        except OSError as e:
            raise ValidationError({field_name: f'Failed to get stat: {e}'})

        size = stat.st_size
        modtime = datetime.fromtimestamp(stat.st_mtime).astimezone()

        errs = {}
        if self.size is None and field_name == 'file_pipeline':
            self.size = size
        elif self.size != size:
            errs['size'] = (
                f'size changed -- expected: {self.size}, actually: {size}'
            )

        if self.modtime is None and field_name == 'file_pipeline':
            self.modtime = modtime
        elif self.modtime != modtime:
            errs['modtime'] = (
                f'modtime changed -- expected (modulo TZ): {self.modtime}, '
                f'actually: {modtime}'
            )
        if errs:
            raise ValidationError(errs)

    def is_public(self):
        """
        Convenience method telling if a file is public or restricted

        Returns True for files belonging to public datasets/samples.

        This may hit the database to retrieve the file's sample.
        """
        return self.sample.is_public()

    def compute_local_path(self):
        """
        Return what the local storage path **should** be.

        Would return empty str for files not to be published via local storage.

        The POLICY is to publish to local storage all files of non-public
        samples and metagenomic assemblies for public samples.  Files keep
        their original name and relative path.
        """
        if self.is_public():
            if self.filetype != self.Type.METAG_ASM:
                return ''
        return self.file_pipeline.name

    def compute_globus_path(self):
        """
        Return what the globus/public path **should** be.

        Would return empty str for files not to be published via globus.

        The current POLICY is to publish all tracked files from a public sample
        under their original name and relative path.
        """
        if self.is_public():
            return self.file_pipeline.name
        else:
            return ''

    def set_file_local(self, save=True):
        """ Automatically set value for file_local field """
        changed = self.update_storage('file_local')
        if changed and save:
            self.save()

    def set_file_globus(self, save=True):
        """ Automatically set value for file_globus field """
        changed = self.update_storage('file_globus')
        if changed and save:
            self.save()

    def update_storage(self, field_name, force_name=None, dry_run=False):
        """
        Ensure file's existence in storage follows policy.

        force_name: Override policy with this name, only for testing/debugging.
        dry_run bool: Only print what would be done, for dev/debug.

        Returns True if anything changed, False otherwise.  The instance will
        not be saved.  Raises exception if underlying storage operation fails.
        In such an error case the field-file's name attribute is not reset and
        may differ from the original and what's stored at the DB.
        """
        if field_name == 'file_local':
            new_name = self.compute_local_path()
        elif field_name == 'file_globus':
            new_name = self.compute_globus_path()
        else:
            raise ValueError('illegal field_name')

        if force_name:
            new_name = force_name

        file = getattr(self, field_name)

        if file.name == new_name:
            # no change
            # TODO: check at least?
            return False

        old_name = file.name

        if file:
            if new_name:
                # move existing file
                file.name = new_name
                if dry_run:
                    print(f'[dryrun] moving ({field_name}) {old_name}->{file}')
                else:
                    file.storage.move(old_name, file)
            else:
                # delete
                if dry_run:
                    print(f'[dryrun] delete ({field_name}) {file}')
                else:
                    file.delete(save=False)
        else:
            # new file
            file.name = new_name
            if dry_run:
                print(f'[dryrun] link/copy ({field_name}) {file}')
            else:
                file.storage.link_or_copy(self.file_pipeline, file)

        if dry_run:
            file.name = old_name

        return True

    @property
    def description(self):
        return f'{self.get_filetype_display()} for {self.sample}'

    @classmethod
    def update_omics_pipeline_checkout(cls, current=None, dry_run=True):
        """
        Compile and save list of timestamped pipeline output files

        This is a utility for development and testing purpose until the omics
        pipeline/snakemake learns how to writes such a file.  The purpose is to
        avoid reading partially written pipeline output files in a robust way
        when running load_omics_data().

        current:
            Path-like to checkout text file.  This file will be appended to if
            not in dry_run mode.

        dry_run:
            If True, then write output to stdout.  If False, then append to the
            current file.  Other values are interpreted as path-like to which
            the new data is written (appended, too.)
        """
        LOCK = settings.OMICS_PIPELINE_ROOT / '.snakemake/locks/0.output.lock'
        Sample = cls._meta.get_field('sample').related_model

        if current is None:
            current = settings.OMICS_CHECKOUT_FILE

        if cls.pipeline_checkout is None:
            cls.load_pipeline_checkout(current)

        # 1. get existing file to mtime mapping
        old_data = cls.pipeline_checkout
        print(f'Previously known files: {len(old_data)}')

        # 2. get omics sample data
        sample_data = []
        with Sample.loader.get_omics_import_file().open() as ifile:
            got_header = False
            for line in ifile:
                row = line.split('\t', maxsplit=5)
                sample_id, _, _, sample_type, sample_dir, *_ = row
                if not got_header:
                    if sample_id != 'SampleID':
                        raise RuntimeError(f'unexpected header in: {line}')
                    if sample_type != 'sample_type':
                        raise RuntimeError(f'unexpected header in: {line}')
                    if sample_dir != 'sample_dir':
                        raise RuntimeError(f'unexpected header in: {line}')
                    got_header = True
                    continue

                if sample_dir.startswith('data/omics/'):
                    sample_dir = sample_dir.removeprefix('data/omics/')
                    sample_data.append((sample_id, sample_type, sample_dir))
        print(f'Samples registered in pipeline: {len(sample_data)}')

        # 3. Read snakemake output lock file
        # These are paths relative to OMICS_PIPELINE_ROOT, keeping those under
        # data root.
        locked_files = set()
        data_root = cls._meta.get_field('file_pipeline').storage.location
        data_pref = data_root.relative_to(settings.OMICS_PIPELINE_ROOT)
        with open(LOCK) as ifile:
            for lineno, line in enumerate(ifile):
                path = Path(line.rstrip('\n'))
                try:
                    # rm data_pref to make these paths comparable with
                    # File.file_pipeline.name
                    path = path.relative_to(data_pref)
                except ValueError:
                    # locked file is not relative to our data root, ignore
                    continue
                locked_files.add(path)
        del data_root, data_pref, path, line, ifile
        print(f'Lock file: {len(locked_files)}/{lineno + 1} files under data '
              f'root')

        # 4. check file stats
        data = {}
        num_same = num_new = num_updated = num_too_recent = num_locked = 0
        now = datetime.now().astimezone()
        one_day = timedelta(days=1)
        warn_os_error = True
        print('Checking file stats... ', end='', flush=True)
        for sid, styp, sdir in sample_data:
            for j in cls.Type:
                sample = Sample(
                    sample_id=sid,
                    sample_type=styp,
                    analysis_dir=sdir,
                )
                try:
                    file_obj = sample.get_omics_file(j)
                except ValueError:
                    if j not in cls.PATH_TAILS:
                        # type is not in File.PATH_TAILS
                        continue
                    else:
                        raise
                file = file_obj.file_pipeline

                try:
                    st = Path(file.path).stat()
                except OSError:
                    # e.g. file not found, permissions
                    continue
                else:
                    warn_os_error = False

                path = file.name
                dt = datetime.fromtimestamp(st.st_mtime).astimezone()

                use_dt = True
                if path in old_data:
                    if path in data:
                        # faulty sample data
                        raise RuntimeError(f'dupe: {path}')

                    if old_data[path] < dt:
                        num_updated += 1
                        print(f'{old_data[path]} -> {dt} {path}',
                              file=sys.stderr)  # DEBUG
                    else:
                        # is up-to-date
                        num_same += 1
                        use_dt = False
                else:
                    num_new += 1

                if (now - dt) < one_day:
                    num_too_recent += 1
                    use_dt = False

                if path in locked_files:
                    num_locked += 1
                    use_dt = False

                if use_dt:
                    data[path] = dt
        print('[done]')

        print(f'up-to-date: {num_same}')
        print(f'newer time: {num_updated}')
        print(f'  new file: {num_new}')
        print(f'too recent: {num_too_recent}')
        print(f'    locked: {num_locked}')

        if sample_data and warn_os_error:
            print('WARNING: no files could be access, check storage mount etc')

        # 4. write out data
        with ExitStack() as estack:
            if dry_run is True:
                outpath = None
                ofile = sys.stdout
            else:
                outpath = current if dry_run is False else dry_run
                ofile = estack.enter_context(open(outpath, 'a'))
                print(f'Saving as {ofile.name} ...', end='', flush=True)

            for path, dt in data.items():
                ofile.write(f'{dt}\t{path}\n')
        if outpath:
            print('[OK]')

    @classmethod
    def load_pipeline_checkout(cls, path=None):
        """
        helper to import the omics pipeline good output files listing

        This sets and populates a dict mapping paths to last modified datetime.
        """
        if path is None:
            path = settings.OMICS_CHECKOUT_FILE
        files = {}
        with open(path) as ifile:
            for line in ifile:
                mtime, _, relpath = line.strip().partition('\t')
                # later entries overwrite earlier
                files[Path(relpath)] = datetime.fromisoformat(mtime)
        cls.pipeline_checkout = files

    def verify_with_pipeline(self):
        """ Verify that modtime matches pipeline checkout time """
        cls = self.__class__
        if cls.pipeline_checkout is None:
            cls.load_pipeline_checkout()
        mtime = cls.pipeline_checkout.get(Path(self.file_pipeline.name), None)
        if mtime is None:
            raise ValidationError({str(self): 'file not in pipeline checkout'})
        if not self.modtime:
            raise ValidationError({str(self): 'modtime not set'})
        if self.modtime != mtime:
            raise ValidationError(
                {str(self): f'modtime {self.modtime} differs (modulo TZ)'
                            f'from pipeline checkout {mtime}'}
            )


class TaxonAbundance(Model):
    """ Abundance of taxon in a sample """
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    # Fields cf. Kraken-style report mmseqs2 userguide
    taxon = models.ForeignKey(
        TaxNode,
        on_delete=models.CASCADE,
        null=True, blank=True,  # null means 'unclassified' / mmseqs2's taxid 0
        related_name='abundance',
    )
    tpm = models.FloatField(verbose_name='TPM')

    objects = managers.TaxonAbundanceManager()
    loader = managers.TaxonAbundanceLoader()

    class Meta(Model.Meta):
        unique_together = (
            ('sample', 'taxon'),
        )

    def __str__(self):
        if self.taxon is None:
            orgname = 'unclassified'
        else:
            orgname = self.taxon.name  # queries DB
        return f'{orgname}/{self.sample.sample_id}: {self.tpm}'


class CompoundAbundance(AbstractAbundance):
    """
    abundance vs. compounds

    with data from Sample_xxxxx_compounds_VERSION.txt files
    """
    ROLE_CHOICES = (
        ('r', 'REACTANT'),
        ('p', 'PRODUCT'),
        ('t', 'TRANSPORT'),
    )
    compound = models.ForeignKey(
        CompoundRecord,
        related_name='abundance',
        **fk_req,
    )
    role = models.CharField(max_length=1, choices=ROLE_CHOICES)

    loader = managers.CompoundAbundanceLoader()

    class Meta(AbstractAbundance.Meta):
        unique_together = (
            ('sample', 'compound', 'role'),
        )

    def __str__(self):
        return f'{self.compound.accession} ({self.role[0]}) {self.rpkm}'


class SequenceLike(Model):
    """
    Abstract model for sequences as found in fasta (or similar) files
    """
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)

    fasta_offset = models.PositiveBigIntegerField(
        **opt,
        help_text='offset of first byte of fasta (or similar) header, if there'
                  ' is one, otherwise first byte of sequence',
    )
    fasta_len = models.PositiveIntegerField(
        **opt,
        help_text='length of sequence record in bytes, header+sequence '
                  'including internal and final newlines or until EOF.'
    )

    objects = managers.SequenceLikeManager()

    class Meta:
        abstract = True

    def get_sequence(self, fasta_format=False, file=None,
                     original_header=False):
        """
        Retrieve sequence from file storage, optionally fasta-formatted

        To get sequences for many objects, use the to_fasta() queryset method,
        which is much more efficient than calling this method while iterating
        over a queryset.

        If fasta_format is True, then the output will be a two-line string that
        may end with a newline.  If original_header is False a new header based
        on the model instance will be generated.  If fasta_format is False, the
        return value will be a string without any newlines.
        """
        if original_header and not fasta_format:
            raise ValueError('incompatible parameters: can only ask for '
                             'original header with fasta format')
        if file is None:
            p = self._meta.managers_map['loader'].get_fasta_path(self.sample)
            fh = p.open('rb')
        else:
            fh = file

        skip_header = not fasta_format or not original_header
        try:
            seq = get_fasta_sequence(fh, self.fasta_offset, self.fasta_len,
                                     skip_header=skip_header)
        finally:
            if file is None:
                fh.close()

        seq = seq.decode()
        if fasta_format and not original_header:
            return ''.join([f'>{self}\n', seq])
        else:
            return seq


class Contig(SequenceLike):
    contig_no = models.PositiveIntegerField()
    # columns from *_contig_abund.tsv file
    mean = models.FloatField(**opt)
    trimmed_mean = models.FloatField(**opt)
    covered_bases = models.PositiveIntegerField(**opt)
    variance = models.FloatField(**opt)
    length = models.PositiveIntegerField(**opt)
    reads = models.PositiveIntegerField(**opt)
    reads_per_base = models.FloatField(**opt)
    rpkm = models.FloatField(**opt)
    tpm = models.FloatField(**opt)
    # lca from *_contig_lca.tsv
    lca = models.ForeignKey(TaxNode, **fk_opt)

    loader = managers.ContigLoader()

    class Meta:
        default_manager_name = 'objects'
        unique_together = (
            ('sample', 'contig_no'),
        )

    def __str__(self):
        return self.accession

    @property
    def accession(self):
        return f'{self.sample.sample_id}_{self.contig_no}'

    def set_from_fa_head(self, fasta_head_line):
        # parsing ">samp_14_123\n" -> 123
        self.contig_no = int(fasta_head_line.strip().split('_')[2])


class FuncAbundance(AbstractAbundance):
    """
    abundance vs functions

    With data from the Sample_xxxx_functions_VERSION.txt files
    """
    function = models.ForeignKey(
        FuncRefDBEntry,
        related_name='abundance',
        **fk_req,
    )

    loader = managers.FuncAbundanceLoader()

    class Meta(Model.Meta):
        unique_together = (
            ('sample', 'function'),
        )

    def genes(self):
        """
        Queryset of associated genes
        """
        return Gene.objects.filter(
            sample=self.sample,
            besthit__function_refs=self.function,
        )


class Gene(Model):
    """ Model for a contig vs. UniRef100 alignment hit """
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    # fields below cf. BLAST output fmt 6 (mmseqs2 tophit_aln)
    contig = models.ForeignKey('Contig', **fk_req)
    ref = models.ForeignKey(UniRef100, **fk_req)
    pident = models.FloatField()
    length = models.PositiveIntegerField()
    mismatch = models.PositiveSmallIntegerField()
    gapopen = models.PositiveSmallIntegerField()
    qstart = models.PositiveSmallIntegerField()
    qend = models.PositiveIntegerField()
    sstart = models.PositiveSmallIntegerField()
    send = models.PositiveIntegerField()
    # skip evalue
    bitscore = models.PositiveSmallIntegerField()
    # ?? abund = models.ForeignKey(Abundance, **fk_opt) (from tophit_report)

    loader = managers.GeneLoader()

    class Meta:
        unique_together = (
            ('sample', 'contig', 'ref', 'qstart', 'qend', 'sstart', 'send'),
        )


class NCRNA(Model):
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    contig = models.ForeignKey('Contig', **fk_req)
    match = models.ForeignKey('RNACentralRep', **fk_req)
    part = models.PositiveIntegerField(**opt)

    # SAM alignment section data
    flag = models.PositiveIntegerField(help_text='bitwise FLAG')
    pos = models.PositiveIntegerField(
        help_text='1-based leftmost mapping position',
    )
    mapq = models.PositiveIntegerField(help_text='MAPing Quality')

    class Meta:
        unique_together = (
            ('sample', 'contig', 'part'),
        )

    def __str__(self):
        if self.part is None:
            part = ''
        else:
            part = f'part_{self.part}'
        return f'{self.sample.accession}:{self.contig}{part}->{self.match}'

    @classmethod
    def get_sam_file(cls, sample):
        return (settings.OMICS_DATA_ROOT / 'NCRNA'
                / f'{sample.accession}_convsrna.sam')


class Protein(SequenceLike):
    gene = models.OneToOneField(Gene, **fk_req)

    def __str__(self):
        return str(self.gene)

    @classmethod
    def have_sample_data(cls, sample, set_to=None):
        if set_to is None:
            return sample.proteins_ok
        else:
            sample.proteins_ok = set_to
            sample.save()

    @classmethod
    def get_fasta_path(cls, sample):
        return (settings.OMICS_DATA_ROOT / 'PROTEINS'
                / f'{sample.accession}_PROTEINS.faa')

    @classmethod
    def get_load_sample_fasta_extra_kw(cls, sample):
        # returns a dict of the sample's genes pks
        qs = sample.gene_set.values_list('gene_id', 'pk')
        return dict(gene_ids=dict(qs.iterator()))

    def set_from_fa_head(self, line, **kwargs):
        if 'gene_ids' in kwargs:
            gene_ids = kwargs['gene_ids']
        else:
            raise ValueError('Expect "contig_ids" in kw args')

        # parsing prodigal info
        gene_accn, _, _ = line.lstrip('>').rstrip().partition(' # ')

        self.gene_id = gene_ids[gene_accn]


class ReadLibrary(Model):
    sample = models.OneToOneField(
        settings.OMICS_SAMPLE_MODEL,
        on_delete=models.CASCADE,
        related_name='reads',
    )
    fwd_qc0_fastq = DataPathField(base='READS')
    rev_qc0_fastq = DataPathField(base='READS')
    fwd_qc1_fastq = DataPathField(base='READS')
    rev_qc1_fastq = DataPathField(base='READS')
    raw_read_count = models.PositiveIntegerField(**opt)
    qc_read_count = models.PositiveIntegerField(**opt)

    class Meta:
        verbose_name_plural = 'read libraries'

    def __str__(self):
        return self.sample.accession

    @classmethod
    @atomic
    def sync(cls, no_counts=True, raise_on_error=False):
        if not no_counts:
            raise NotImplementedError('read counting is not yet implemented')

        for i in get_sample_model().objects.filter(reads=None):
            obj = cls.from_sample(i)
            try:
                obj.full_clean()
            except ValidationError as e:
                if raise_on_error:
                    raise
                else:
                    log.error(f'failed validation: {repr(obj)}: {e}')
                    continue
            obj.save()
            log.info(f'new read lib: {obj}')

    @classmethod
    def from_sample(cls, sample):
        obj = cls()
        obj.sample = sample
        fq = sample.get_fq_paths()
        obj.fwd_qc0_fastq = fq['dd_fwd']
        obj.rev_qc0_fastq = fq['dd_rev']
        obj.fwd_qc1_fastq = fq['ddtrnhnp_fwd']
        obj.rev_qc1_fastq = fq['ddtrnhnp_rev']
        return obj


class RNACentral(Model):
    history = None
    RNA_TYPES = (
        # unique 3rd column from rnacentral_ids.txt
        (1, 'antisense_RNA'),
        (2, 'autocatalytically_spliced_intron'),
        (3, 'guide_RNA'),
        (4, 'hammerhead_ribozyme'),
        (5, 'lncRNA'),
        (6, 'miRNA'),
        (7, 'misc_RNA'),
        (8, 'ncRNA'),
        (9, 'other'),
        (10, 'piRNA'),
        (11, 'precursor_RNA'),
        (12, 'pre_miRNA'),
        (13, 'ribozyme'),
        (14, 'RNase_MRP_RNA'),
        (15, 'RNase_P_RNA'),
        (16, 'rRNA'),
        (17, 'scaRNA'),
        (18, 'scRNA'),
        (19, 'siRNA'),
        (20, 'snoRNA'),
        (21, 'snRNA'),
        (22, 'sRNA'),
        (23, 'SRP_RNA'),
        (24, 'telomerase_RNA'),
        (25, 'tmRNA'),
        (26, 'tRNA'),
        (27, 'vault_RNA'),
        (28, 'Y_RNA'),
    )
    INPUT_FILE = (settings.OMICS_PIPELINE_DATA / 'NCRNA' / 'RNA_CENTRAL'
                  / 'rnacentral_clean.fasta.gz')

    accession = AccessionField()
    taxon = models.ForeignKey(TaxNode, **fk_req)
    rna_type = models.PositiveSmallIntegerField(choices=RNA_TYPES)

    def __str__(self):
        return (f'{self.accession} {self.taxon.taxid} '
                f'{self.get_rna_type_display()}')

    @classmethod
    def load(cls, path=INPUT_FILE):
        type_map = dict(((b.casefold(), a) for a, b, in cls.RNA_TYPES))
        taxa = dict(TaxNode.objects.values_list('taxid', 'pk'))

        zcat_cmd = ['/usr/bin/unpigz', '-c', str(path)]
        zcat = Popen(zcat_cmd, stdout=PIPE)

        pp = ProgressPrinter('rna central records read')
        objs = []
        try:
            print('BORK STAGE I')
            for line in zcat.stdout:
                # line is bytes
                if not line.startswith(b'>'):
                    continue
                line = line.decode()
                accn, typ, taxid2, _ = line.lstrip('>').split('|', maxsplit=4)
                # taxid2 can be multiple, so take taxid from accn (ask Teal?)
                accn, _, taxid = accn.partition('_')
                objs.append(cls(
                    accession=accn,
                    taxon_id=taxa[int(taxid)],
                    rna_type=type_map[typ.lower()],
                ))
                pp.inc()
        except Exception:
            raise
        finally:
            try:
                zcat.communicate(timeout=15)
            except TimeoutExpired:
                zcat.kill()
                zcat.communicate()
            if zcat.returncode:
                log.error(f'{zcat_cmd} returned with status {zcat.returncode}')

        pp.finish()
        log.info(f'Saving {len(objs)} RNA Central accessions...')
        cls.objects.bulk_create(objs)


class RNACentralRep(Model):
    """ Unique RNACentral representatives """
    history = None


class SampleTracking(Model):
    """
    Track progress loading omics data for a sample
    """
    class Flag(models.TextChoices):
        METADATA = 'MD', 'meta data loaded'
        PIPELINE = 'PL', 'omics pipeline registered'
        ASSEMBLY = 'ASM', 'assembly loaded'
        UR1ABUND = 'UAB', 'reads/UR100 abundance loaded'
        UR1TPM = 'TPM', 'reads/UR100/TPM loaded'
        TAXABUND = 'TAB', 'taxa abundance loaded'
        CABUND = 'CAB', 'contig abundance loaded'

    flag = models.CharField(max_length=3, choices=Flag.choices)
    sample = models.ForeignKey(
        settings.OMICS_SAMPLE_MODEL,
        on_delete=models.CASCADE,
        related_name='tracking',
    )
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    info = models.JSONField(blank=True, default=dict)

    objects = managers.SampleTrackingManager()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['flag', 'sample'],
                name='uniq_samptrack_flag',
            ),
        ]

    def __str__(self):
        return f'{self.sample.sample_id}/{self.flag}'

    @property
    def job(self):
        jobcls = type(self).objects.job_classes[self.flag]
        # FIXME: the returned job may come from the job class-level cache and
        # then may have a different tracking instance. This may just be ugly
        # but I should re-think the surrounding design .
        return jobcls.for_sample(self.sample, tracking=self)

    def undo(self, fake=False):
        """
        Unload the data loaded by our job and delete this tracking item.

        fake bool:
            If True then the job's undo function will not be run.  Use with
            care.  The default is False.
        """
        if fake:
            self.job.fake_undo()
        else:
            self.job.run_undo()


class Sample(AbstractSample):
    """
    Placeholder model for samples
    """
    class Meta:
        swappable = 'OMICS_SAMPLE_MODEL'


class Dataset(AbstractDataset):
    """
    Placeholder model implementing a dataset
    """
    class Meta:
        swappable = 'OMICS_DATASET_MODEL'
