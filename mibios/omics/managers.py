"""
Module for data load managers
"""

from collections import defaultdict
from io import StringIO
from itertools import groupby, islice
from logging import getLogger
import os
import shutil
import subprocess
import tempfile

from django.conf import settings
from django.db import connection
from django.db.transaction import atomic
from django.utils.module_loading import import_string

from mibios.models import QuerySet
from mibios.ncbi_taxonomy.models import DeletedNode, MergedNodes, TaxNode
from mibios.umrad.models import UniRef100
from mibios.umrad.manager import BulkLoader, Manager, MetaDataLoader
from mibios.umrad.utils import CSV_Spec, atomic_dry, InputFileError

from . import get_sample_model
from .utils import call_each, get_fasta_sequence

log = getLogger(__name__)


# FIXME move function to good home
def resolve_glob(path, pat):
    value = None
    for i in path.glob(pat):
        if value is None:
            return i
        else:
            raise RuntimeError(f'glob is not unique: {i}')
    raise FileNotFoundError(f'glob does not resolve: {path / pat}')


class SampleLoadMixin:
    """ Mixin for Loader class that loads per-sample files """

    load_flag_attr = None
    """ may be specified by implementing class """

    sample = None
    """ sample is set by load_sample() for use in per-field helper methods """

    @atomic_dry
    def load_sample(self, sample, template=None, sample_filter=None,
                    done_ok=True, redo=False, undo=False, **kwargs):

        if undo and redo:
            raise ValueError('Option undo is incompatible with option redo.')

        if undo and kwargs.get('update'):
            raise ValueError('Option undo is incompatible with option update.')

        if template is None:
            template = {'sample': sample}

        if sample_filter is None:
            sample_filter = template

        if 'flag' in kwargs:
            flag = kwargs.pop('flag')
            if flag is None:
                # explicit override / no flag check/set
                pass
        else:
            flag = self.load_flag_attr

        if flag:
            update = kwargs.get('update', False)
            done = getattr(sample, flag)
            if done and done_ok and not redo:
                # nothing to do
                return

            if done and not done_ok:
                raise RuntimeError(f'already loaded: {flag}->{sample}')

            if done and redo and not update:
                # have to delete records for sample
                self.undo(sample, sample_filter, flag)

        if 'file' not in kwargs:
            kwargs.update(file=self.get_file(sample))

        self.sample = sample
        self.load(template=template, **kwargs)
        # ensure subsequent calls of manager methods never get wrong sample:
        self.sample = None

        if flag:
            setattr(sample, flag, True)
            sample.save()

    @atomic
    def unload_sample(self, sample, sample_filter=None, flag=None):
        """
        Delete all objects related to the given sample

        This is to undo the effect of load_sample().  Override this method if a
        more delicate operation is needed.
        """
        if flag is None:
            flag = getattr(self, 'load_flag_attr')
        if flag is None:
            raise ValueError('load progress flag needs to be provided')

        if sample_filter is None:
            sample_filter = {'sample': sample}

        print('Deleting... ', end='', flush=True)
        dels = self.model.objects.filter(**sample_filter).delete()
        print(dels, '[OK]')

        setattr(sample, flag, False)
        sample.save()


class TaxNodeMixin:
    """
    Loader mixin for input files with NCBI taxid column

    Use together with SampleLoadMixin.
    """
    def check_taxid(self, value, obj):
        """
        Check validity of NCBI taxids

        Preprocessing method for taxid columns

        Add this method to spec line for NCBI taxid columns.  Checks if taxid
        was merged and returns the new taxid instead.  Returns None for deleted
        taxids.  Returns None if value is 0, meaning 'unclassified' in mmseqs2
        semantics.  Otherwise unknown taxids are let through unchanged.
        """
        if value == '0':
            # mmseq2 may assign taxid 0 / no rank / unclassified
            return None
        if int(value) in self.deleted:
            self.deleted_count += 1
            return None
        if new_taxid := self.merged.get(int(value), None):
            self.merged_count += 1
            return new_taxid
        return value

    @atomic_dry
    def load_sample(self, sample, **kwargs):
        print('Retrieving merged/deleted taxid info... ', end='', flush=True)
        qs = MergedNodes.objects.select_related('new_node__taxid')
        qs = qs.values_list('old_taxid', 'new_node__taxid')
        self.merged = dict(qs)
        self.merged_count = 0
        self.deleted = set(DeletedNode.objects.values_list('taxid', flat=True))
        self.deleted_count = 0
        print(f'[{len(self.merged)}/{len(self.deleted)} OK]')
        super().load_sample(sample, **kwargs)
        if self.merged_count:
            print(f'Found {self.merged_count} rows with outdated but merged '
                  f'taxids.')
        if self.deleted_count:
            print(f'Found {self.deleted_count} rows with deleted taxids. '
                  f'TaxNode field set to None.')
        del self.merged, self.merged_count, self.deleted, self.deleted_count


class CompoundAbundanceLoader(BulkLoader, SampleLoadMixin):
    """ loader manager for CompoundAbundance """
    load_flag_attr = 'comp_abund_ok'

    spec = CSV_Spec(
        ('cpd', 'compound'),
        ('cpd_name', None),
        ('type', 'role'),
        ('CPD_GENES', None),
        ('CPD_SCO', 'scos'),
        ('CPD_RPKM', 'rpkm'),
        ('cpdlca', None),
    )

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.sample_id}_compounds_*.txt'
        )


class SequenceLikeQuerySet(QuerySet):
    """ objects manager for sequence-like models """

    def to_fasta(self):
        """
        Make fasta-formatted sequences
        """
        files = {}
        lines = []
        fields = ('fasta_offset', 'fasta_len', 'gene_id', 'sample__accession')
        qs = self.select_related('sample').values_list(*fields)
        try:
            for offs, length, gene_id, sampid in qs.iterator():
                if sampid not in files:
                    sample = get_sample_model().objects.get(accession=sampid)
                    files[sampid] = \
                        self.model.loader.get_fasta_path(sample).open('rb')

                lines.append(f'>{sampid}:{gene_id}')
                lines.append(
                    get_fasta_sequence(files[sampid], offs, length).decode()
                )
        finally:
            for i in files.values():
                i.close()

        return '\n'.join(lines)


SequenceLikeManager = Manager.from_queryset(SequenceLikeQuerySet)


class SequenceLikeLoader(SampleLoadMixin, BulkLoader):
    """
    Loader manager for the SequenceLike abstract model

    Provides the load_fasta() method.
    """

    def get_fasta_path(self, sample):
        """
        return path to fasta file that contains our sequence
        """
        # must be implemented by inheriting class
        # should return Path
        raise NotImplementedError

    def get_set_from_fa_head_extra_kw(self, sample):
        """
        Return extra kwargs for from_fasta()

        Should be overwritten by inheriting class if needed
        """
        return {}

    @atomic_dry
    def load_fasta(self, sample, start=0, limit=None, bulk=True,
                   validate=False, done_ok=True, redo=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        done = getattr(sample, self.fasta_load_flag)
        if done:
            if done_ok:
                if redo:
                    self.unload_fasta_sample()
                else:
                    # nothing to do
                    return
            else:
                raise RuntimeError(
                    f'data already loaded - update not supported: '
                    f'{self.fasta_load_flag} -> {sample}'
                )

        objs = self.from_fasta(sample, start=start, limit=limit)
        if validate:
            objs = call_each(objs, 'full_clean')

        if bulk:
            self.bulk_create(objs)
        else:
            for i in objs:
                i.save()

        setattr(sample, self.fasta_load_flag, True)
        sample.save()

    @atomic
    def unload_fasta(self, sample):
        """
        Counterpart of load_fasta but will delete more, abundance etc.
        """
        self.unload_sample(sample, flag=self.fasta_load_flag)

    def from_fasta(self, sample, start=0, limit=None):
        """
        Generate instances for given sample

        Helper for load_fasta().
        """
        extra = self.get_set_from_fa_head_extra_kw(sample)
        with self.get_fasta_path(sample).open('r') as fa:
            os.posix_fadvise(fa.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            print(f'reading {fa.name} ...')
            obj = None
            data = self._fasta_parser(fa)
            for header, offs, byte_len in islice(data, start, limit):
                obj = self.model(
                    sample=sample,
                    fasta_offset=offs,
                    fasta_len=byte_len,
                )
                try:
                    obj.set_from_fa_head(header, **extra)
                except Exception as e:
                    raise RuntimeError(
                        f'failed parsing fa head in file {fa.name}: '
                        f'{e.__class__.__name__}: {e}:{header}'
                    ) from e
                yield obj

    def _fasta_parser(self, file):
        """ helper to iterate over fasta record infos """
        header = None
        record_offs = None
        while True:
            pos = file.tell()
            line = file.readline()
            if line.startswith('>') or not line:
                if header is not None:
                    yield header, record_offs, pos - record_offs
                header = line.lstrip('>').rstrip()
                record_offs = pos

            if not line:
                # EOF
                break


class ContigLoader(TaxNodeMixin, SequenceLikeLoader):
    """ Manager for the Contig model """
    fasta_load_flag = 'contig_fasta_loaded'

    def get_fasta_path(self, sample):
        return sample.get_metagenome_path() / 'assembly' \
            / 'megahit_noNORM' / 'final.contigs.renamed.fa'

    def get_contig_abund_path(self, sample):
        """ get path to samp_NNN_contig_abund.tsv file """
        fname = f'{sample.sample_id}_contig_abund.tsv'
        return sample.get_metagenome_path() / fname

    def get_contig_lca_path(self, sample):
        """ get path to samp_NNN_contig_lca.tsv file """
        fname = f'{sample.sample_id}_contig_lca.tsv'
        return sample.get_metagenome_path() / fname

    def get_contig_no(self, value, obj):
        sample_id, _, contig_no = value.rpartition('_')
        if sample_id != self.sample.sample_id:
            raise InputFileError(
                f'bad sample_id in contig name: {value} but expected '
                f'{self.sample.sample_id}'
            )
        try:
            contig_no = int(contig_no)
        except (ValueError, TypeError) as e:
            raise InputFileError(f'bad contig number in {value=}: {e}') from e
        return contig_no

    contig_abund_spec = CSV_Spec(
        ('Sample', CSV_Spec.IGNORE_COLUMN),
        ('Contig', 'contig_no', get_contig_no),
        ('Mean', 'mean'),
        ('Trimmed Mean', 'trimmed_mean'),
        ('Covered Bases', 'covered_bases'),
        ('Variance', 'variance'),
        ('Length', 'length'),
        ('Read Count', 'reads'),
        ('Reads per base', 'reads_per_base'),
        ('RPKM', 'rpkm'),
        ('TPM', 'tpm'),
    )

    contig_lca_spec = CSV_Spec(
        ('contig_no', get_contig_no),
        ('lca', 'check_taxid'),
    )

    @atomic_dry
    def unload_fasta_sample(self, sample):
        super().unload_fasta_sample(sample)
        # since this deletes the whole object
        sample.contig_abundance_loaded = False
        sample.contig_lca_loaded = False
        sample.save()

    @atomic_dry
    def load_abundance(self, sample, **kwargs):
        self.load_sample(
            sample,
            flag='contig_abundance_loaded',
            spec=self.contig_abund_spec,
            file=self.get_contig_abund_path(sample),
            update=True,
            **kwargs)

    @atomic_dry
    def unload_abundance(self, sample):
        fields = ['mean', 'trimmed_mean', 'covered_bases', 'variance',
                  'length', 'reads', 'reads_per_base', 'rpkm', 'tpm']
        print('Unsetting abundance fields... ', end='', flush=True)
        count = self.filter(sample=sample).update(**{i: None for i in fields})
        print(f'[{count} OK]')
        sample.contig_abundance_loaded = False
        sample.save()

    @atomic_dry
    def load_lca(self, sample, **kwargs):
        self.load_sample(
            sample,
            flag='contig_lca_loaded',
            spec=self.contig_lca_spec,
            file=self.get_contig_lca_path(sample),
            update=True,
            **kwargs,
        )

    @atomic_dry
    def unload_lca(self, sample):
        print('Unsetting lca... ', end='', flush=True)
        count = self.filter(sample=sample).update(lca=None)
        print(f'[{count} OK]')
        setattr(sample, 'contig_lca_loaded', False)
        sample.save()


class FuncAbundanceLoader(BulkLoader, SampleLoadMixin):
    load_flag_attr = 'func_abund_ok'

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.sample_id}_functionss_*.txt'
        )


class GeneLoader(SampleLoadMixin, BulkLoader):
    def get_alignment_file(self, sample):
        return sample.get_metagenome_path() \
            / f'{sample.sample_id}_contig_tophit_aln'

    def to_contig_id(self, value, obj):
        """ return ID tuple based on sample and contig number """
        return self.sample.pk, int(value.rpartition('_')[2])

    def parse_ur100(self, value, obj):
        # UniRef100_ --> UNIREF100_
        return value.upper()

    spec = CSV_Spec(
        ('contig', to_contig_id),
        ('ref', parse_ur100),
        ('pident', ),
        ('length', ),
        ('mismatch', ),
        ('gapopen', ),
        ('qstart', ),
        ('qend', ),
        ('sstart', ),
        ('send', ),
        (None, ),  # skip evalue column
        ('bitscore', ),
    )

    @atomic_dry
    def load_alignments(self, sample, **kwargs):
        """
        Load data from contig_tophits_aln file

        Creates Gene records.
        """
        infile = self.get_alignment_file(sample)

        # Ensure only relevant ur100s are retrieved by load() later:
        print(f'Extracting distinct unirefs from {infile}...', end='',
              flush=True)
        with infile.open() as ifile:
            # get 2nd column w/o prefix, unique values only
            urefs = {
                line.split(maxsplit=2)[1].removeprefix('UniRef100_')
                for line in ifile
            }
        if connection.vendor == 'sqlite':
            # avoid "too many SQL variables" kicking in when len(urefs)>250000
            def get_urefs_map():
                urefs_map = UniRef100.loader.only('accession') \
                                     .in_bulk(urefs, field_name='accession')
                return (((accn, obj.pk) for accn, obj in urefs_map.items()))

            self.spec.fkmap_filters['ref'] = get_urefs_map
        else:
            self.spec.fkmap_filters['ref'] = {'accession__in': urefs}
        print(f' [{len(urefs)} OK]')

        dedup = StringIO()
        ur100s = set()
        with infile.open() as ifile:
            print(f'Reading file: {ifile.name} ...', end='', flush=True)
            totalcount = 0
            rows = ((i.rstrip('\n').split('\t') for i in ifile))
            # sort+group by Gene unique_together fields
            key = lambda x: (x[0], x[1], x[6], x[7], x[8], x[9])  # noqa:E731
            rows = sorted(rows, key=key)
            for count, (_, grp) in enumerate(groupby(rows, key=key)):
                grp = list(grp)
                totalcount += len(grp)
                row = max(grp, key=lambda x: int(x[-1]))  # max bitscore
                ur100s.add(self.parse_ur100(row[1], None))
                dedup.write('\t'.join(row))
                dedup.write('\n')
        dedup.seek(0)
        if count == totalcount:
            print(f' [{count} OK]')
        else:
            print(f'deduplicated: {totalcount} -> {count} rows [OK]')
        dedup.seek(0)

        # Create placeholders for missing UR100 records
        print('Check for missing UniRef100s... ', end='', flush=True)
        existing = UniRef100.objects.in_bulk(ur100s, field_name='accession')
        missing = ur100s.difference(existing.keys())
        if missing:
            print(f'{len(missing)} missing out of {len(ur100s)}')
            objs = ((UniRef100(accession=i) for i in missing))
            UniRef100.loader.bulk_create(objs)
        else:
            print('[all good]')
            objs = None
        del ur100s, existing, missing, objs

        self.load_sample(
            sample,
            flag='gene_alignments_loaded',
            file=dedup,
            **kwargs,
        )


class SampleLoader(MetaDataLoader):
    """ Loader manager for Sample """
    def get_omics_import_file(self):
        """ get the omics data import log """
        return settings.OMICS_DATA_ROOT / 'data' / 'imported_samples.tsv'

    @atomic_dry
    def update_analysis_status(self, source_file=None, skip_on_error=False,
                               quiet=False):
        """
        Update sample table with analysis status
        """
        # Input file columns of interest:
        SAMPLE_ID = 0
        STUDY_ID = 1
        TYPE = 3
        ANALYSIS_DIR = 4
        SUCCESS = 6

        COLUMN_NAMES = (
            (SAMPLE_ID, 'SampleID'),
            (STUDY_ID, 'StudyID'),
            (TYPE, 'sample_type'),
            (ANALYSIS_DIR, 'sample_dir'),
            (SUCCESS, 'import_success'),
        )

        def err(msg):
            msg = f'line {lineno}: {msg}'
            if skip_on_error:
                log.error(msg)
            else:
                raise RuntimeError(msg)

        objs = self.select_related('dataset').in_bulk(field_name='sample_id')

        if source_file is None:
            source_file = self.get_omics_import_file()

        with open(source_file) as f:
            print(f'Reading {source_file} ...')
            head = f.readline().rstrip('\n').split('\t')
            for index, column in COLUMN_NAMES:
                if head[index] != column:
                    raise RuntimeError(
                        f'unexpected header in {f.name}: 0-based column '
                        f'{index} is "{head[index]}" but expected "{column}"'
                    )

            good_seen = []
            samp_id_seen = set()
            changed = 0
            unchanged = 0
            notfound = 0
            nosuccess = 0
            for lineno, line in enumerate(f, start=1):
                row = line.rstrip('\n').split('\t')
                sample_id = row[SAMPLE_ID]
                dataset = row[STUDY_ID]
                sample_type = row[TYPE]
                analysis_dir = row[ANALYSIS_DIR]
                success = row[SUCCESS]

                if not all([sample_id, dataset, sample_type, analysis_dir, success]):  # noqa: E501
                    err(f'field is empty in row: {row}')
                    continue

                if sample_id in samp_id_seen:
                    err(f'Duplicate sample ID: {sample_id}')
                    continue
                else:
                    samp_id_seen.add(sample_id)

                if success != 'TRUE':
                    if not quiet:
                        log.info(f'ignoring {sample_id}: no import success')
                    nosuccess += 1
                    continue

                try:
                    obj = objs[sample_id]
                except KeyError:
                    log.warning(f'line {lineno}: unknown sample: {sample_id} '
                                f'(skipping)')
                    notfound += 1
                    continue

                if obj.dataset.dataset_id != dataset:
                    err(
                        f'line {lineno}: {obj} dataset changed: '
                        f'{obj.dataset.dataset_id} -> {dataset}'
                    )
                    continue

                if obj.sample_type != sample_type:
                    err(
                        f'line {lineno}: {obj} sample type changed: '
                        f'{obj.sample_type} -> {sample_type}'
                    )
                    continue

                updateable = ['analysis_dir']
                change_set = []
                for attr in updateable:
                    val = locals()[attr]
                    if getattr(obj, attr) != val:
                        setattr(obj, attr, val)
                        change_set.append(attr)
                if change_set:
                    need_save = True
                    save_info = (f'update: {obj} change_set: '
                                 f'{", ".join(change_set)}')
                else:
                    need_save = False
                    unchanged += 1

                if need_save:
                    obj.metag_pipeline_reg = True
                    obj.full_clean()
                    obj.save()
                    log.info(save_info)
                    changed += 1

                good_seen.append(obj.pk)

        log.info('Summary:')
        log.info(f'  records read from file: {lineno}')
        log.info(f'  (unique) samples listed: {len(samp_id_seen)}')
        log.info(f'  samples updated: {changed}')
        if notfound:
            log.warning(
                f'Samples missing from database (or hidden): {notfound}'
            )

        if nosuccess:
            log.warning(f'Samples not marked "import_success": {nosuccess}')

        if unchanged:
            log.info(f'Data for {unchanged} listed samples remain unchanged.')

        missing_or_bad = self.exclude(pk__in=good_seen)
        if missing_or_bad.exists():
            log.warning(f'The DB has {missing_or_bad.count()} samples '
                        f'which are missing from {source_file} or which had '
                        f'to be skipped for other reasons.')

        missing = self.exclude(sample_id__in=samp_id_seen)
        if missing.exists():
            log.warning(f'The DB has {missing.count()} samples not at all '
                        f'listed in {source_file}')

    def status(self):
        if not self.exists():
            print('No samples in database yet.')
            return

        print(' ' * 10, 'contigs', 'bins', 'checkm', 'genes', sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                'OK' if i.contigs_ok else '',
                'OK' if i.binning_ok else '',
                'OK' if i.checkm_ok else '',
                'OK' if i.genes_ok else '',
                sep='\t'
            )

    def status_long(self):
        if not self.exists():
            print('No samples in database yet')
            return

        print(' ' * 10, 'cont cl', 'MAX', 'MET93', 'MET97', 'MET99', 'genes',
              sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                i.contig_set.count(),
                i.binmax_set.count(),
                i.binmet93_set.count(),
                i.binmet97_set.count(),
                i.binmet99_set.count(),
                i.gene_set.count(),
                sep='\t'
            )

    def get_metagenomic_loader_script(self):
        """
        Gets 'script' to load metagenomic data

        This is a list of pairs (progress_flag, functions) where functions
        means either a single callable that takes a sample as argument or a
        list of such callables.
        """
        Contig = import_string('mibios.omics.models.Contig')
        Gene = import_string('mibios.omics.models.Gene')
        TaxonAbundance = import_string('mibios.omics.models.TaxonAbundance')

        return [
            ('contig_fasta_loaded', Contig.loader.load_fasta),
            ('contig_abundance_loaded', Contig.loader.load_abundance),
            ('contig_lca_loaded', Contig.loader.load_lca),
            ('gene_alignments_loaded', Gene.loader.load_alignments),
            ('tax_abund_ok', TaxonAbundance.loader.load_sample),
        ]

    def get_blocklist(self):
        """
        Return QuerySet of blocked samples

        Blocked samples are those for which something is wrong with the omics
        data.  Beyond meta-data no other data should be loaded.
        """
        if not settings.SAMPLE_BLOCKLIST:
            return self.none()

        blocklist = []
        with open(settings.SAMPLE_BLOCKLIST) as ifile:
            for line in ifile:
                line = line.strip()
                if line.startswith('#'):
                    continue
                blocklist.append(line)

        # assumes the block list has valid sample_id values, if not this will
        # fail silently
        return self.filter(sample_id__in=blocklist)


class TaxonAbundanceLoader(TaxNodeMixin, SampleLoadMixin, BulkLoader):
    """ loader manager for the TaxonAbundance model """
    load_flag_attr = 'tax_abund_ok'

    def process_tpm(self, value, obj):
        # obj.taxon may be None for unclassified or deleted taxids
        # tpm values for those get added together
        taxid = getattr(obj.taxon, 'taxid', None)
        if taxid in self.tpm_data:
            # is a row for merged taxid (duplicate)
            retval = self.spec.SKIP_ROW
        else:
            # zero as temp placeholder
            retval = 0.0

        # save value for post-processing
        self.tpm_data[taxid] += float(value)
        return retval

    spec = CSV_Spec(
        ('tax_id', 'taxon', 'check_taxid'),
        ('abund_w_subtax', None),
        ('abund_direct', 'tpm', process_tpm),
    )

    def get_file(self, sample):
        """ Get path to lca_abund_summarized file """
        if sample.sample_id == 'samp_447':
            from pathlib import Path
            return Path('/tmp/heinro/samp_447_lca_abund_summarized.tsv')
        return sample.get_metagenome_path() \
            / f'{sample.sample_id}_lca_abund_summarized.tsv'

    @atomic_dry
    def load_sample(self, sample, **kwargs):
        """
        Load abundance vs. taxa from lca_abund_summarized files

        The input data may be based on an older version of NCBI's taxonomy.  In
        such a case, abundance for deleted taxids is added to 'unclassified.'
        Merged taxids' abundance is accumulated approriately.  Abundance for
        any otherwise unknown taxids, e.g. if a newer version of NCBI taxonomy
        was used, is ignored (rows will be skipped.)
        """
        self.tpm_data = defaultdict(float)
        super().load_sample(sample, **kwargs)
        tpm_data = self.tpm_data
        del self.tpm_data
        # post-processing: add tpm to parentage
        # 0. Save unclassified
        if None in tpm_data:
            unclass = self.model.objects.get(sample=sample, taxon=None)
            unclass.tpm = tpm_data.pop(None)
            unclass.save()
            print(f'Updated "unclassified" [{unclass.tpm} OK]')
        # 1. close under ancestry
        print(f'Fill-in taxonomic relations {len(tpm_data)} -> ', end='',
              flush=True)
        qs = TaxNode.objects.prefetch_related('ancestors', 'children')
        qs = qs.select_related('parent')
        closure = qs.in_bulk(tpm_data.keys(), field_name='taxid')
        new_taxids = []
        for taxid in list(tpm_data.keys()):
            for a_node in closure[taxid].ancestors.all():
                if a_node.taxid not in closure:
                    closure[a_node.taxid] = a_node
                    new_taxids.append(a_node.taxid)
                    # 2. add ancestors to tpm data
                    tpm_data[a_node.taxid] = 0.0
        print(f'{len(tpm_data)} [OK]')

        # 3. push tpm numbers up the tree
        print('Closing TPM under ancestry... ', end='', flush=True)
        to_update = {}
        to_create = {}
        while tpm_data:
            leaves = []
            for taxid, tpm in tpm_data.items():
                node = closure[taxid]
                is_leaf = True
                for child in node.children.all():
                    if child.taxid in tpm_data:
                        if child.is_root():
                            # Strangely root is it's own parent
                            # averting infinite loop here
                            continue
                        is_leaf = False
                        break
                if is_leaf:
                    if not node.is_root():
                        # root's parent is root, all accumulation is done
                        # already
                        tpm_data[node.parent.taxid] += tpm
                    if taxid in new_taxids:
                        to_create[taxid] = tpm
                    else:
                        to_update[taxid] = tpm
                    leaves.append(taxid)
            for i in leaves:
                del tpm_data[i]
        print(f'[{len(to_update)}/{len(to_create)} OK]')

        # 4. update existing objects
        objs = []
        for obj in self.filter(sample=sample).select_related('taxon'):
            # objs = qs.in_bulk(to_update.keys(), field_name='taxon__taxid')
            if obj.taxon is None:
                continue
            if obj.taxon.taxid in to_update:
                obj.tpm = to_update[obj.taxon.taxid]
                objs.append(obj)
        self.fast_bulk_update(objs, ['tpm'])

        # 5. create new objects (ancestry)
        objs = [
            self.model(sample=sample, taxon=closure[taxid], tpm=tpm)
            for taxid, tpm in to_create.items()
        ]
        self.bulk_create(objs)


class TaxonAbundanceManager(Manager):
    def as_krona_input_text(self, sample, field_name):
        """
        Generate input text data for krona

        Will return an empty string if no abundance data is saved for the given
        sample.
        """
        # field_name may be user input from http requests, double-check that
        # this is a real field, so we don't send garbage to krona (via the
        # getattr below)
        self.model._meta.get_field(field_name)  # may raise FieldDoesNotExist

        qs = self.filter(sample=sample).select_related('taxon')
        qs = qs.prefetch_related('taxon__ancestors')

        rows = []
        for i in qs.iterator():
            lin = sorted(i.taxon.ancestors.all(), key=lambda x: x.rank)
            row = [str(getattr(i, field_name))] + [i.name for i in lin]
            rows.append('\t'.join(row))
        return '\n'.join(rows)

    def as_krona_html(self, sample, field, outpath=None):
        """
        Generate the krona html file

        If outpath is given this is save to the file system, if it is None, the
        default, then Krona's html content is the return value (indended to be
        used by a view).  In the latter case a None return value indicates
        either an error with running the krona generator or missing abundance
        data (Check the django error log for details.)
        """
        # TODO: caching, have a settings option to store krona's static files
        # (CSS, ...) locally instead of at sourceforge or wherever.
        with tempfile.TemporaryDirectory() as tmpd:
            krona_in = tmpd + '/data.txt'
            krona_out = tmpd + '/krona.html'

            with open(krona_in, 'w') as ofile:
                txt = self.as_krona_input_text(sample, field)
                if txt:
                    ofile.write(txt)
                else:
                    # no abundance data for this sample.  If we were to
                    # continue, krona will happily make a page, which would be
                    # mostly empty, will little indication whats going on.
                    if outpath is None:
                        # assume the calling view handles errors
                        log.error(
                            'no tax abundance data, not generating krona page'
                        )
                        return None
                    else:
                        raise ValueError('no abundance data')

            cmd = [
                'ktImportText',
                '-n', f'all of {sample}',
                '-o', krona_out,
                krona_in
            ]
            try:
                subprocess.run(cmd, cwd=tmpd, check=True)
            except subprocess.CalledProcessError as e:
                # e.g. krona not installed
                if outpath is None:
                    raise
                else:
                    log.error(f'krona failed: {e}')
                    return None

            if outpath is None:
                with open(krona_out) as ifile:
                    return ifile.read()
            else:
                shutil.copy2(krona_out, outpath)
