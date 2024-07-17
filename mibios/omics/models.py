from datetime import datetime
from logging import getLogger
from pathlib import Path
import re
from subprocess import PIPE, Popen, TimeoutExpired
from time import time

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.transaction import atomic

from mibios.data import TableConfig
from mibios.ncbi_taxonomy.models import TaxNode
from mibios.umrad.fields import AccessionField, PathField, PathPrefixValidator
from mibios.umrad.model_utils import (
    digits, opt, ch_opt, fk_req, fk_opt, uniq_opt, Model,
)
from mibios.umrad.models import CompoundRecord, FuncRefDBEntry, UniRef100
from mibios.umrad.manager import Manager
from mibios.umrad.utils import ProgressPrinter

from . import managers, get_sample_model, sra
from .amplicon import get_target_genes, quick_analysis, quick_annotation
from .fields import DataPathField
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
        help_text='sample ID or name as given by study',
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
    sra_accession = models.TextField(max_length=16, **ch_opt, verbose_name='SRA accession')  # noqa: E501
    amplicon_target = models.TextField(max_length=16, **ch_opt)
    fwd_primer = models.TextField(max_length=32, **ch_opt)
    rev_primer = models.TextField(max_length=32, **ch_opt)

    # sample data accounting flags
    contig_abundance_loaded = models.BooleanField(
        default=False,
        help_text='contig abundance/rpkm data loaded',
    )
    contig_lca_loaded = models.BooleanField(
        default=False,
        help_text='contig LCA data loaded',
    )
    gene_alignments_loaded = models.BooleanField(
        default=False,
        help_text='genes loaded via contig_tophit_aln file',
    )

    analysis_dir = models.TextField(
        **opt,
        help_text='path to results of analysis, relative to OMICS_DATA_ROOT',
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

    def get_samp_no(self):
        """ Get sample_id number: "samp_NNN" -> NNN """
        return int(self.sample_id.removeprefix('samp_'))

    def load_bins(self):
        if not self.binning_ok:
            with atomic():
                Bin.import_sample_bins(self)
                self.binning_ok = True
                self.save()
        if self.binning_ok and not self.checkm_ok:
            with atomic():
                CheckM.import_sample(self)
                self.checkm_ok = True
                self.save()

    @atomic
    def delete_bins(self):
        with atomic():
            qslist = [self.binmax_set, self.binmet93_set, self.binmet97_set,
                      self.binmet99_set]
            for qs in qslist:
                print(f'{self}: deleting {qs.model.method} bins ...', end='',
                      flush=True)
                counts = qs.all().delete()
                print('\b\b\bOK', counts)
            self.binning_ok = False
            self.checkm_ok = False  # was cascade-deleted
            self.save()

    def get_metagenome_path(self):
        """
        Get path to data analysis / pipeline results

        DEPRECATED - use get_omics_file()
        """
        path = settings.OMICS_DATA_ROOT / 'data' / 'omics' / 'metagenomes' \
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
        base = settings.OMICS_DATA_ROOT / 'READS'
        fname = f'{self.accession}_{{infix}}.fastq.gz'
        return {
            infix: base / fname.format(infix=infix)
            for infix in ['dd_fwd', 'dd_rev', 'ddtrnhnp_fwd', 'ddtrnhnp_rev']
        }

    def get_checkm_stats_path(self):
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
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    number = models.PositiveIntegerField()
    checkm = models.OneToOneField('CheckM', **fk_opt)

    method = None  # set by inheriting model

    class Meta:
        abstract = True
        unique_together = (
            ('sample', 'number'),
        )

    def __str__(self):
        return f'{self.sample.accession} {self.method} #{self.number}'

    @classmethod
    def get_concrete_models(cls):
        """
        Return list of all concrete bin sub-classes/models
        """
        # The recursion stops at the first non-abstract models, so this may not
        # make sense in a multi-table inheritance setting.
        children = cls.__subclasses__()
        if children:
            ret = []
            for i in children:
                ret += i.get_concrete_models()
            return ret
        else:
            # method called on a non-parent
            if cls._meta.abstract:
                return []
            else:
                return [cls]

    @classmethod
    def get_class(cls, method):
        """
        Get the concrete model for give binning method
        """
        if cls.method is not None:
            return super().get_class(method)

        for i in cls.get_concrete_models():
            if i.method == method:
                return i

        raise ValueError(f'not a valid binning type/method: {method}')

    @classmethod
    def import_all(cls):
        """
        Import all binning data

        This class method can be called on the abstract parent Bin class and
        will then import data for all binning types.  Or it can be called on an
        concrete model/class and then will only import data for the
        corresponding binning type.
        """
        if not cls._meta.abstract:
            raise RuntimeError(
                'method can not be called by concrete bin subclass'
            )
        for i in get_sample_model().objects.all():
            cls.import_sample_bins(i)

    @classmethod
    def import_sample_bins(cls, sample):
        """
        Import all types of bins for given sample
        """
        if sample.binning_ok:
            log.info(f'{sample} has bins loaded already')
            return

        if cls._meta.abstract:
            # Bin parent class only
            with atomic():
                noerr = True
                for klass in cls.get_concrete_models():
                    res = klass.import_sample_bins(sample)
                    noerr = bool(res) and noerr
                if noerr:
                    sample.binning_ok = True
                    sample.save()
                return

        # Bin subclasses only
        files = list(cls.bin_files(sample))
        if not files:
            log.warning(f'no {cls.method} bins found for {sample}')
            return None

        for i in sorted(files):
            res = cls._import_bins_file(sample, i)
            if not res:
                log.warning(f'got empty cluster from {i} ??')

        return len(files)

    @classmethod
    def _import_bins_file(cls, sample, path):
        _, num, _ = path.name.split('.')
        try:
            num = int(num)
        except ValueError as e:
            raise RuntimeError(f'Failed parsing filename {path}: {e}')

        obj = cls.objects.create(sample=sample, number=num)
        cids = []
        with path.open() as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                cids.append(line.strip().lstrip('>'))

        qs = Contig.objects.filter(sample=sample, contig_id__in=cids)
        kwargs = {cls._meta.get_field('members').remote_field.name: obj}
        qs.update(**kwargs)
        log.info(f'{obj} imported: {len(cids)} contigs')
        return len(cids)


class BinMAX(Bin):
    method = 'MAX'

    @classmethod
    def bin_files(cls, sample):
        """
        Generator over bin file paths
        """
        pat = f'{sample.accession}_{cls.method}_bins.*.fasta'
        path = settings.OMICS_DATA_ROOT / 'BINS' / 'MAX_BIN'
        return path.glob(pat)

    class Meta:
        verbose_name = 'MaxBin'
        verbose_name_plural = 'MaxBin bins'


class BinMetaBat(Bin):

    class Meta(Bin.Meta):
        abstract = True

    @classmethod
    def bin_files(cls, sample):
        """
        Generator over bin file paths
        """
        pat = f'{sample.accession}_{cls.method}_bins.*'
        path = settings.OMICS_DATA_ROOT / 'BINS' / 'METABAT'
        return path.glob(pat)


class BinMET93(BinMetaBat):
    method = 'MET_P97S93E300'

    class Meta:
        verbose_name = 'MetaBin 97/93'
        verbose_name_plural = 'MetaBin 97/93 bins'


class BinMET97(BinMetaBat):
    method = 'MET_P99S97E300'

    class Meta:
        verbose_name = 'MetaBin 99/97'
        verbose_name_plural = 'MetaBin 99/97 bins'


class BinMET99(BinMetaBat):
    method = 'MET_P99S99E300'

    class Meta:
        verbose_name = 'MetaBin 99/99'
        verbose_name_plural = 'MetaBin 99/99 bins'


class CheckM(Model):
    """
    CheckM results for a bin
    """
    history = None
    translation_table = models.PositiveSmallIntegerField(
        verbose_name='Translation table',
    )
    gc_std = models.FloatField(verbose_name='GC std')
    ambiguous_bases = models.PositiveIntegerField(
        verbose_name='# ambiguous bases',
    )
    genome_size = models.PositiveIntegerField(verbose_name='Genome size')
    longest_contig = models.PositiveIntegerField(verbose_name='Longest contig')
    n50_scaffolds = models.PositiveIntegerField(verbose_name='N50 (scaffolds)')
    mean_scaffold_len = models.FloatField(verbose_name='Mean scaffold length')
    num_contigs = models.PositiveIntegerField(verbose_name='# contigs')
    num_scaffolds = models.PositiveIntegerField(verbose_name='# scaffolds')
    num_predicted_genes = models.PositiveIntegerField(
        verbose_name='# predicted genes',
    )
    longest_scaffold = models.PositiveIntegerField(
        verbose_name='Longest scaffold',
    )
    gc = models.FloatField(verbose_name='GC')
    n50_contigs = models.PositiveIntegerField(verbose_name='N50 (contigs)')
    coding_density = models.FloatField(verbose_name='Coding density')
    mean_contig_length = models.FloatField(verbose_name='Mean contig length')

    class Meta:
        verbose_name = 'CheckM'
        verbose_name_plural = 'CheckM records'

    @classmethod
    def import_all(cls):
        for i in get_sample_model().objects.all():
            if i.checkm_ok:
                log.info(f'sample {i}: have checkm stats, skipping')
                continue
            cls.import_sample(i)

    @classmethod
    @atomic
    def import_sample(cls, sample):
        bins = {}
        stats_file = sample.get_checkm_stats_path()
        if not stats_file.exists():
            log.warning(f'{sample}: checkm stats do not exist: {stats_file}')
            return

        for binid, obj in cls.from_file(stats_file):
            # parse binid, is like: Sample_42895_MET_P99S99E300_bins.6
            part1, _, num = binid.partition('.')  # separate number part
            parts = part1.split('_')
            sample_name = '_'.join(parts[:2])
            meth = '_'.join(parts[2:-1])  # rest but without the "_bins"

            try:
                num = int(num)
            except ValueError as e:
                raise ValueError(f'Bad bin id in stats: {binid}, {e}')

            if sample_name != sample.accession:
                raise ValueError(
                    f'Bad sample name in stats: {binid} -- expected: '
                    f'{sample.accession}'
                )

            try:
                binclass = Bin.get_class(meth)
            except ValueError as e:
                raise ValueError(f'Bad method in stats: {binid}: {e}') from e

            try:
                binobj = binclass.objects.get(sample=sample, number=num)
            except binclass.DoesNotExist as e:
                raise RuntimeError(
                    f'no bin with checkm bin id: {binid} file: {stats_file}'
                ) from e

            binobj.checkm = obj

            if binclass not in bins:
                bins[binclass] = []

            bins[binclass].append(binobj)

        for binclass, bobjs in bins.items():
            binclass.objects.bulk_update(bobjs, ['checkm'])

        sample.checkm_ok = True
        sample.save()

    @classmethod
    def from_file(cls, path):
        """
        Create instances from given bin_stats.analyze.tsv file.

        Should normally be only called from CheckM.import_sample().
        """
        ret = []
        with path.open() as fh:
            for line in fh:
                bin_key, data = line.strip().split('\t')
                data = data.lstrip('{').rstrip('}').split(', ')
                obj = cls()
                for item in data:
                    key, value = item.split(': ', maxsplit=2)
                    key = key.strip("'")
                    for i in cls._meta.get_fields():
                        if i.is_relation:
                            continue
                        # assumes correclty assigned verbose names
                        if i.verbose_name == key:
                            field = i
                            break
                    else:
                        raise RuntimeError(
                            f'Failed parsing {path}: no matching field for '
                            f'{key}, offending line is:\n{line}'
                        )

                    try:
                        value = field.to_python(value)
                    except ValidationError as e:
                        message = (f'Failed parsing field "{key}" on line:\n'
                                   f'{line}\n{e.message}')
                        raise ValidationError(
                            message,
                            code=e.code,
                            params=e.params,
                        ) from e

                    setattr(obj, field.attname, value)

                obj.save()
                ret.append((bin_key, obj))

        return ret


class File(Model):
    """ An omics product file, analysis pipeline result """

    def get_path_prefix():
        return settings.OMICS_DATA_ROOT / 'data' / 'omics'

    def get_public_prefix():
        return settings.PUBLIC_DATA_ROOT

    class Type(models.IntegerChoices):
        METAG_ASM = 1, 'metagenomic assembly, fasta format'
        """ e.g. samp_14/assembly/megahit_noNORM/final.contigs.renamed.fa """
        METAT_ASM = 2, 'metatranscriptome assembly, fasta format'
        """ TODO """
        FUNC_ABUND = 3, 'functional abundance, csv format'
        """ e.g. samp_14/samp_14_tophit_report """
        TAX_ABUND = 4, 'taxonomic abundance, csv format'

    path = PathField(
        root=get_path_prefix,
        unique=True,
    )
    public = PathField(
        root=get_public_prefix,
        blank=True,
        # null corresponds to file not published
        null=True,
    )
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
                fields=['public'],
                condition=~models.Q(public=None),  # blank is okay
                name='%(app_label)s_%(class)s_public_path_unique',
            ),
        ]

    PATH_TAILS = {
        Type.METAG_ASM:
            Path('assembly', 'megahit_noNORM', 'final.contigs.renamed.fa'),
        Type.TAX_ABUND: '{sample.sample_id}_lca_abund_summarized.tsv',
        Type.FUNC_ABUND: '{sample.sample_id}_tophit_report',
    }
    """ file location, relative to a sample's data dir """

    def __str__(self):
        return f'{self.path}'

    def __fspath__(self):
        """ implement os.PathLike """
        return str(self.path)

    _stat = None
    """ attribute holding cached stat_result, cf. File.stat() """

    def stat(self, from_cache=True):
        """
        Get stat for file under path and cache the result
        """
        if not from_cache or self._stat is None:
            self._stat = self.path.stat()
        return self._stat

    path_validator = PathPrefixValidator(settings.OMICS_DATA_ROOT / 'data' / 'omics')  # noqa:E501
    public_validator = PathPrefixValidator(settings.PUBLIC_DATA_ROOT)  # noqa:E501

    def full_clean(self):
        # Auto set size, modtime.  This needs to happen before running
        # full_clean() and needs a valid path, so that's validated first (and
        # then later in super().full_clean() again).  Also prefix-validate
        # path, public at instance level with validators outside of the field
        # to avoid leaking settings into migrations.
        errors = {}

        try:
            path = self._meta.get_field('path').clean(self.path, self)
        except ValidationError as e:
            errors = e.update_error_dict(errors)
        else:
            # tests that only make sense if path is valid:
            try:
                self.path_validator(path)
            except ValidationError as e:
                errors = e.update_error_dict(errors)

            # Usually self._stat is not populated yet, or we want to re-check a
            # file pulled from the DB, so run stat() here on the cleaned path
            # (as self.path may be relative, etc.)
            try:
                self._stat = path.stat()
            except OSError as e:
                ve = ValidationError({'path': f'stat call failed: {e}'})
                errors = ve.update_error_dict(errors)
            else:
                size = self.stat().st_size
                modtime = datetime.fromtimestamp(self.stat().st_mtime) \
                                  .astimezone()
                if self.pk is None:
                    self.size = size
                    self.modtime = modtime
                else:
                    errs = {}
                    if self.size != size:
                        errs['size'] = (
                            f'size changed -- expected: {self.size}, actually:'
                            f' {size}'
                        )

                    if self.modtime != modtime:
                        errs['modtime'] = (
                            f'modtime changed -- expected: {self.modtime}, '
                            f'actually: {modtime}'
                        )
                    if errs:
                        ve = ValidationError(errs)
                        errors = ve.update_error_dict(errors)

        try:
            public = self._meta.get_field('public').clean(self.public, self)
        except ValidationError as e:
            errors = e.update_error_dict(errors)
        else:
            if public:
                try:
                    self.public_validator(public)
                except ValidationError as e:
                    errors = e.update_error_dict(errors)

        try:
            super().full_clean()
        except ValidationError as e:
            errors = e.update_error_dict(errors)

        if errors:
            raise ValidationError(errors)

    @property
    def root(self):
        return self._meta.get_field('path').root

    @property
    def public_root(self):
        return self._meta.get_field('public').root

    @property
    def relpath(self):
        """ The path relative to the common path prefix """
        return self.path.relative_to(self.root)

    @property
    def relpublic(self):
        """
        The public path relative to the common path prefix

        This property may be exposed on a public page.

        Returns None if no public path is set or if the common prefix is not
        configured.
        """
        if self.public_root is None or self.public is None:
            return None
        return self.public.relative_to(self.public_root)

    def compute_public_path(self):
        """
        Return what the public path **should** be.

        Would return None for files not to be published.

        The hereby implemented POLICY is to publish all tracked files under
        their original name and relative path.
        """
        # Uncomment this to unpublish all files
        # return None
        return self.public_root / self.relpath

    def manage_public_path(self):
        """
        Manage the public path

        This method will set the public field according to policy.  The policy
        itself is not implemented in here but via compute_public_path().  This
        method will ensure the published file exist in the public space in the
        filesystem.  It will set the public field, the file instance is not
        saved however, the caller has to save it.

        Returns a status tuple of booleans indicating any change of the public
        field value: (old path unlinked, new path set).  A status of (True,
        True) means an existing public path got changed.
        """
        old = self.public
        self.public = self.compute_public_path()
        if self.public == old:
            if self.public:
                if self.public.is_file():
                    # all good
                    pass
                else:
                    # should be a rare case
                    print(f'[WARNING] fix missing file: {self.public}', end='', flush=True)  # noqa:E501
                    self._hardlink()
                    print('[OK]')
            else:
                # file remains unpublished
                pass
            # no changes
            return (False, False)
        elif old is None:
            self._hardlink()
            # a new File
            return (False, True)
        else:
            # e.g. policy change
            if old.is_file():
                self._unlink(old)
            else:
                print(f'[NOTICE] file already gone: {old}')
            if self.public:
                self._hardlink()
                # change
                return (True, True)
            else:
                # unpublished
                return (True, False)

    def _hardlink(self):
        """ helper to hardlink public file against internal """
        public = self.public
        target = self.path
        if 'public' not in public.parts and 'public-test' not in public.parts:
            # some safety guard
            raise RuntimeError('unsave operation suspected')

        if not public.is_relative_to(self.public_root):
            # another safety guard
            raise RuntimeError('unsave operation suspected')

        public.parent.mkdir(parents=True, exist_ok=True)
        public.hardlink_to(target)

    def _unlink(self, oldpath):
        """
        Helper to remove a file from the public space

        Cleans up any resulting empty directories.
        """
        # some rail guards:
        if not oldpath.is_relative_to(self.public_root):
            raise RuntimeError('unsave operation suspected')
        if oldpath == self.public:
            raise RuntimeError('unsave operation suspected')

        oldpath.unlink()
        parent = oldpath.parent
        while parent.is_relative_to(self.public_root):
            if parent == self.public_root:
                # keep the root
                break
            try:
                parent.rmdir()
            except OSError:
                # e.g. is not empty
                break

            parent = parent.parent

    @property
    def download_url(self):
        """ direct download URLs """
        if settings.GLOBUS_DIRECT_URL_BASE and self.relpublic:
            # cf. https://docs.globus.org/globus-connect-server/v5.4/https-access-collections/  # noqa:E501
            base = settings.GLOBUS_DIRECT_URL_BASE.rstrip('/')
            return f'{base}/{self.relpublic}?download'
        else:
            return None

    @property
    def description(self):
        return f'{self.get_filetype_display()} for {self.sample}'

    @classmethod
    def make_omics_pipeline_checkout(cls, outpath=None):
        """
        Compile and save list of timestamped pipeline output files

        This is a utility for development and testing purpose.
        """
        root = cls.get_path_prefix()
        Sample = cls._meta.get_field('sample').related_model
        with open(outpath, 'w') as ofile:
            for i in Sample.loader.exclude(analysis_dir=None):
                for j in cls.Type:
                    try:
                        path = i.get_omics_file(j).path
                    except ValueError:
                        if j not in cls.PATH_TAILS:
                            # type is not in File.PATH_TAILS
                            continue
                        else:
                            raise

                    try:
                        st = path.stat()
                    except OSError:
                        # e.g. file not found
                        continue

                    relpath = path.relative_to(root)
                    dt = datetime.fromtimestamp(st.st_mtime).astimezone()
                    ofile.write(f'{dt}\t{relpath}\n')

    @classmethod
    def load_pipeline_checkout(cls, path='omics.checkout.txt'):
        """
        helper to import the omics pipeline good output files listing

        This sets and populates a dict mapping paths to last modified datetime.
        """
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
        mtime = cls.pipeline_checkout.get(self.relpath, None)
        if mtime is None:
            raise ValidationError('file not in pipeline checkout')
        if not self.modtime:
            raise ValidationError(f'modtime not set: {self}')
        if self.modtime != mtime:
            raise ValidationError(f'{self}: modtime {self.modtime} differs '
                                  f'from pipeline checkout {mtime}')


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

    # bin membership
    bin_max = models.ForeignKey(BinMAX, **fk_opt, related_name='members')
    bin_m93 = models.ForeignKey(BinMET93, **fk_opt, related_name='members')
    bin_m97 = models.ForeignKey(BinMET97, **fk_opt, related_name='members')
    bin_m99 = models.ForeignKey(BinMET99, **fk_opt, related_name='members')

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
    INPUT_FILE = (settings.OMICS_DATA_ROOT / 'NCRNA' / 'RNA_CENTRAL'
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
        TAXABUND = 'TAB', 'taxa abundance loaded'

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
