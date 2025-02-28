"""
Module for data load managers
"""
from collections import defaultdict
from contextlib import ExitStack
from datetime import date
from functools import cached_property, partial
from itertools import groupby, islice
from logging import getLogger
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from django.conf import settings
from django.db.transaction import atomic, set_rollback
from django.utils.module_loading import import_string

from mibios import __version__ as version
from mibios.models import QuerySet
from mibios.ncbi_taxonomy.models import (
    DeletedNode, MergedNodes, TaxNode, TaxName,
)
from mibios.umrad.models import UniRef100
from mibios.umrad.manager import BulkLoader, Manager, MetaDataLoader
from mibios.umrad.utils import CSV_Spec, atomic_dry, InputFileError, SkipRow

from . import get_sample_model
from .utils import (call_each, gentle_int, get_fasta_sequence,
                    get_sample_blocklist, Timestamper)

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

    sample = None
    """ sample is set by load_sample() for use in per-field helper methods """

    @atomic_dry
    def load_sample(self, sample, template=None, sample_filter=None, **kwargs):
        if template is None:
            template = {'sample': sample}

        if sample_filter is None:
            sample_filter = template

        if 'file' not in kwargs:
            kwargs.update(file=self.get_file(sample))

        self.sample = sample
        self.load(template=template, **kwargs)
        # ensure subsequent calls of manager methods never get wrong sample:
        # FIXME: this is a stupid design
        self.sample = None

    @atomic_dry
    def unload_sample(self, sample, sample_filter=None):
        """
        Delete all objects related to the given sample

        This is to undo the effect of load_sample().  Override this method if a
        more delicate operation is needed.
        """
        if sample_filter is None:
            sample_filter = {'sample': sample}

        print('Deleting... ', end='', flush=True)
        dels = self.model.objects.filter(**sample_filter).delete()
        print(dels, '[OK]')


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
            # FIXME: use get_omics_file() API
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
    def load_fasta(self, sample, start=0, limit=None, file=None, bulk=True,
                   validate=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        objs = self.from_fasta(sample, start=start, limit=limit, file=file)
        if validate:
            objs = call_each(objs, 'full_clean')

        if bulk:
            self.bulk_create(objs)
        else:
            for i in objs:
                i.save()

    @atomic
    def unload_fasta(self, sample):
        """
        Counterpart of load_fasta but will delete more, abundance etc.
        """
        self.unload_sample(sample)

    def from_fasta(self, sample, file=None, start=0, limit=None):
        """
        Generate instances for given sample

        Helper for load_fasta().
        """
        if file is None:
            file = self.get_fasta_path(sample)
        extra = self.get_set_from_fa_head_extra_kw(sample)
        with open(file, 'r') as fa:
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
    def get_fasta_path(self, sample):
        return sample.get_omics_file('METAG_ASM').path

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


fkmap_cache = {}


def fkmap_cache_reset():
    """ Initialize or reset the global fkmap cache """
    global fkmap_cache
    fkmap_cache = {'UniRef100': {}}


class UniRefMixin:
    """ Mixin for dealing with uniref100 columns """
    def parse_ur100(self, value, obj):
        """ Preprocessing method, add this to spec line """
        # UniRef100_XYZ --> XYZ
        return UniRef100.loader.parse_ur100(value)

    def uniref100_helper(self, field_name, spec=None, create_missing=True):
        """
        Helper to run before calling load()

        Reads the uniref100 column and provides the fkmap filter.  For missing
        UniRef100 records, placeholders with only the accession are created.

        :param spec:
            Alternative loader spec attribute.  By default the usual
            Loader.spec is used.
        """
        if spec is None:
            spec = self.spec

        rows = self.spec.iterrows()
        print('Extracting distinct UniRef100s...', end='', flush=True)
        # get accessions w/o prefix, unique values only
        urefs = {
            UniRef100.loader.parse_ur100(
                spec.row2dict(spec.row_data(row))[field_name]
            )
            for row in rows
        }
        print(f' [{len(urefs)} OK]')

        # Create placeholders for missing UR100 records ...
        print('Check for missing UniRef100s... ', end='', flush=True)
        # 1. Look in cache (if initialized) and load any not yet in cache
        global fkmap_cache
        fkmap = fkmap_cache.get('UniRef100', dict())
        missing = urefs.difference((i[0] for i in fkmap.keys()))
        # 2. update cache
        exist_missing = UniRef100.loader.pkmap('accession', list(missing))
        add_to_cache_count = 0
        for accn, pk in exist_missing.items():
            fkmap[(accn,)] = pk
            missing.remove(accn)
            add_to_cache_count += 1
        if add_to_cache_count and fkmap_cache:
            print(f'(cache: +{add_to_cache_count}) ', end='', flush=True)

        if missing:
            print(f'missing: {len(missing)}')
            if create_missing:
                objs = ((UniRef100(accession=i) for i in missing))
                UniRef100.loader.bulk_create(objs)
                # Update fkmap with PKs of new objects:
                fkmap.update((
                    ((accn,), pk) for accn, pk
                    in UniRef100.loader.pkmap_vals('accession', list(missing))
                ))
        else:
            print('[all good]')
            objs = None

        # set fkmap_filter
        self.spec.fkmap_filters[field_name] = lambda: fkmap


class GeneLoader(UniRefMixin, SampleLoadMixin, BulkLoader):
    def get_file(self, sample):
        return sample.get_metagenome_path() \
            / f'{sample.sample_id}_contig_tophit_aln'

    def to_contig_id(self, value, obj):
        """ return ID tuple based on sample and contig number """
        return self.sample.pk, int(value.rpartition('_')[2])

    spec = CSV_Spec(
        ('contig', to_contig_id),
        ('ref', 'parse_ur100'),
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

    def dedup_alns(self):
        """
        deduplicate alignments before loading

        Creates a temporary file and set the spec up to use that as input file.
        """
        dedup = tempfile.SpooledTemporaryFile(mode='w+t')
        totalcount = 0

        rows = self.spec.iterrows()  # iterator over original input file
        # sort+group by Gene unique_together fields
        key = lambda x: (x[0], x[1], x[6], x[7], x[8], x[9])  # noqa:E731
        rows = sorted(rows, key=key)
        print('Deduplicating alignments... ', end='', flush=True)
        for count, (_, grp) in enumerate(groupby(rows, key=key)):
            grp = list(grp)
            totalcount += len(grp)
            row = max(grp, key=lambda x: int(x[-1]))  # max bitscore
            dedup.write('\t'.join(row))
            dedup.write('\n')
        if count == totalcount:
            print(f'[{count} OK]')
        else:
            print(f'[{totalcount} -> {count} OK]')

        # supplant input file
        dedup.seek(0)
        self.spec.file = dedup

    @atomic_dry
    def load_alignments(self, sample, **kwargs):
        """
        Load data from contig_tophits_aln file

        Creates Gene records.  The main entrypoint for the GeneLoader.
        """
        self.spec.pre_load_hook = [
            partial(self.dedup_alns),
            partial(self.uniref100_helper, field_name='ref'),
        ]
        self.spec.fkmap_filters['contig'] = {'sample': sample}

        self.load_sample(
            sample,
            flag='gene_alignments_loaded',
            **kwargs,
        )


class GeneAbundanceLoader(UniRefMixin, SampleLoadMixin, BulkLoader):
    """ load data from contig_tophit_report files """
    load_flag_attr = 'gene_abundance_loaded'
    spec = CSV_Spec(
        ('ref', 'parse_ur100'),
        (None, ),  # ignore the count
        ('unique_cov', ),
        ('target_cov', ),
        ('avg_ident', ),
        # ignore taxonomy columns
        has_header=False,
    )

    def get_file(self, sample):
        return sample.get_metagenome_path() \
            / f'{sample.sample_id}_contig_tophit_report'

    @atomic_dry
    def load_sample(self, sample, *args, **kwargs):
        self.spec.pre_load_hook = \
            partial(self.uniref100_helper, field_name='ref')
        super().load_sample(sample, *args, **kwargs)


class ReadAbundanceLoader(UniRefMixin, SampleLoadMixin, BulkLoader):
    """ load data from *_tophit_report files """

    report_spec = CSV_Spec(
        ('ref', 'parse_ur100'),
        ('read_count', ),
        ('unique_cov', ),
        ('target_cov', ),
        ('avg_ident', ),
        # ignore taxonomy columns
        has_header=False,
    )

    tpm_spec = CSV_Spec(
        ('uniref100_id', 'ref', 'parse_ur100'),
        ('tpm', 'tpm'),
        ('rpkm', 'rpkm'),
    )

    def get_file(self, sample):
        """ get the *_tophit_report file """
        return sample.get_omics_file('FUNC_ABUND')

    @atomic_dry
    def load_sample(self, sample, *args, spec=None, **kwargs):
        if spec is None:
            self.spec = self.report_spec
        else:
            self.spec = spec

        self.spec.pre_load_hook = \
            partial(self.uniref100_helper, field_name='ref')
        super().load_sample(sample, *args, **kwargs)

    @atomic_dry
    def load_tpm_sample(self, sample, *args, file=None, spec=None, **kwargs):
        """
        Load tpm, rpkm values from tophit_TPM files.  Run after load_samples()
        """
        if spec is None:
            self.spec = self.tpm_spec
        else:
            self.spec = spec

        self.spec.pre_load_hook = \
            partial(self.uniref100_helper, field_name='ref')
        if file is None:
            file = sample.get_omics_file('FUNC_ABUND_TPM')
        update = kwargs.pop('update', True)
        if not update:
            raise ValueError('update kwarg must not be False, loader method '
                             'must run in update mode')
        super().load_sample(sample, *args, file=file, update=True, **kwargs)

    @atomic_dry
    def unload_tpm_sample(self, sample):
        num = self.filter(sample=sample).update(tpm=None, rpkm=None)
        print(f'{sample.sample_id}: tpm+rpkm erased for {num} '
              f'{self.model._meta.model_name}')


class SampleLoader(MetaDataLoader):
    """ Loader manager for Sample """

    @classmethod
    def get_omics_import_file(cls):
        """ get the omics data import log """
        basedir = settings.OMICS_PIPELINE_ROOT / 'data' / 'import_logs'
        # log file name begins with date YYYYMMDD:
        basename = '_sample_status.tsv'
        most_recent_date = ''
        most_recent_file = None
        for i in basedir.glob(f'*{basename}'):
            date = i.name.removesuffix(basename)
            if date.isnumeric() and len(date) == 8 and date > most_recent_date:
                most_recent_date = date
                most_recent_file = i
        if most_recent_file is None:
            raise RuntimeError('sample status file / import log not found')
        else:
            return most_recent_file

    @atomic_dry
    def update_from_pipeline_registry(
        self, source_file=None, skip_on_error=False, quiet=False
    ):
        """
        Update sample table with pipeline import status
        """
        SampleTracking = import_string('mibios.omics.models.SampleTracking')

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

        # ExitStack: To cleanly close log file (if needed) and input file as
        # there is some interaction with the atomic_dry wrapper
        with ExitStack() as estack:

            # setup log file output
            if log_dir := getattr(settings, 'IMPORT_LOG_DIR', ''):
                # find a unique log file name and set up the log handler
                log_file_base = Path(log_dir) / 'omics_status_update'
                today = date.today()
                suffix = f'.{today}.log'
                num = 0
                while (log_file := log_file_base.with_suffix(suffix)).exists():
                    num += 1
                    suffix = f'.{today}.{num}.log'

                log_file = estack.enter_context(log_file.open('w'))
                print(f'Logging to: {log_file.name}')
            else:
                log_file = sys.stdout

            def log(msg):
                if quiet and msg.startswith('INFO'):
                    return
                print(msg, file=log_file)

            def err(msg):
                msg = f'ERROR: {msg}'
                if skip_on_error:
                    log(msg)
                else:
                    raise RuntimeError(msg)

            objs = self.select_related('dataset') \
                       .in_bulk(field_name='sample_id')

            if source_file is None:
                source_file = self.get_omics_import_file()

            print(f'Reading {source_file} ...')
            srcf = estack.enter_context(open(source_file))
            head = srcf.readline().rstrip('\n').split('\t')
            for index, column in COLUMN_NAMES:
                if head[index] != column:
                    raise RuntimeError(
                        f'unexpected header in {srcf.name}: 0-based column '
                        f'{index} is "{head[index]}" but expected "{column}"'
                    )

            good_seen = []
            samp_id_seen = set()
            changed = 0
            unchanged = 0
            notfound = 0
            nosuccess = 0
            for lineno, line in enumerate(srcf, start=1):
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
                        log(f'INFO ignoring {sample_id}: no import success')
                    nosuccess += 1
                    continue

                try:
                    obj = objs[sample_id]
                except KeyError:
                    log(f'WARNING line {lineno}: unknown sample: {sample_id} '
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

                analysis_dir = analysis_dir.removeprefix('data/omics/')

                updateable = ['analysis_dir']
                change_set = []
                for attr in updateable:
                    val = locals()[attr]
                    if getattr(obj, attr) != val:
                        setattr(obj, attr, val)
                        change_set.append(attr)
                if change_set:
                    need_save = True
                    save_info = (f'INFO update: {obj} change_set: '
                                 f'{", ".join(change_set)}')
                else:
                    need_save = False
                    unchanged += 1

                if need_save:
                    obj.full_clean()
                    obj.save()
                    log(save_info)
                    changed += 1

                good_seen.append(obj.pk)

                tr, new = SampleTracking.objects.get_or_create(
                    sample=obj,
                    flag=SampleTracking.Flag.PIPELINE,
                )
                if not new:
                    tr.save()  # update timestamp

            srcf.close()

            log('Summary:')
            log(f'  records read from file: {lineno}')
            log(f'  (unique) samples listed: {len(samp_id_seen)}')
            log(f'  samples updated: {changed}')
            if notfound:
                log(f'WARNING Samples missing from database (or hidden): '
                    f'{notfound}')

            if nosuccess:
                log(f'WARNING Samples not marked "import_success": '
                    f'{nosuccess}')

            if unchanged:
                log(f'Data for {unchanged} listed samples remain unchanged.')

            missing_or_bad = self.exclude(pk__in=good_seen)
            if missing_or_bad.exists():
                log(f'WARNING The DB has {missing_or_bad.count()} samples '
                    f'which are missing from {source_file} or which had '
                    f'to be skipped for other reasons.')

            missing = self.exclude(sample_id__in=samp_id_seen)
            if missing.exists():
                log(f'WARNING The DB has {missing.count()} samples not at all '
                    f'listed in {source_file}')

    def get_metagenomic_loader_script(self):
        """
        Gets 'script' to load metagenomic data

        This is a list of pairs (progress_flag, functions) where functions
        means either a single callable that takes a sample as argument or a
        list of such callables.
        """
        Contig = import_string('mibios.omics.models.Contig')
        Gene = import_string('mibios.omics.models.Gene')

        return [
            ('contig_abundance_loaded', Contig.loader.load_abundance),
            ('contig_lca_loaded', Contig.loader.load_lca),
            ('gene_alignments_loaded', Gene.loader.load_alignments),
        ]

    def get_omics_blocklist(self):
        """
        Return QuerySet of samples for which 'omics data loading is blocked

        Blocked samples are those for which something is wrong with the omics
        data.
        """
        blocklist = []
        for sample_id, fields in get_sample_blocklist().items():
            if not fields or 'omics' in fields:
                blocklist.append(sample_id)

        # assumes the block list has valid sample_id values, if not this will
        # fail silently
        return self.filter(sample_id__in=blocklist)

    @gentle_int
    def load_omics_data(self, jobs=None, samples=None, dry_run=False):
        """
        Run given jobs or ready jobs for given samples

        Usually called via Sample.loader.all().load_omics_data()

        This is a wrapper to handle the dry_run parameter.  In a dry run
        everything is done inside an outer transaction that is then rolled
        back, as usual.  But in a production run we don't want the outer
        transaction as to not lose work of successful jobs when we later crash.
        """
        if dry_run:
            with atomic():
                ret = self._load_omics_data(jobs=jobs, samples=samples)
                set_rollback(True)
                return ret
        else:
            return self._load_omics_data(jobs=jobs, samples=samples)

    def _load_omics_data(self, jobs=None, samples=None):
        if jobs and samples:
            raise ValueError('either provide jobs or sample, not both')

        if not jobs and not samples:
            raise ValueError('either jobs or sample must be provided')

        timestamper = Timestamper(
            template='[ {timestamp} ]  ',
            file_copy=settings.OMICS_LOADING_LOG,
        )
        with timestamper:
            print(f'Loading omics data / version: {version}')
            if samples:
                if isinstance(samples, self._queryset_class):
                    sample_qs = samples
                else:
                    sample_qs = self.filter(pk__in=[i.pk for i in samples])
                jobs = sample_qs.get_ready(sort_by_sample=True)
                if not jobs:
                    print('NOTICE: Given samples have no ready jobs.')

            jobs_total = len(jobs)
            # some of the accounting below makes more sense if jobs are sorted
            # by sample, but the logic should still work even if later jobs
            # return to same sample.
            jobs_per_sample = [
                (sample, list(job_grp))
                for sample, job_grp
                in groupby(jobs, key=lambda x: x.sample)
            ]
            sample_count = len(set((i for i, _ in jobs_per_sample)))
            if samples:
                if no_job_sample_count := len(samples) - sample_count:
                    print(f'{no_job_sample_count} of the given samples have no'
                          f' jobs ready')
            print(f'{jobs_total} jobs over {sample_count} samples are ready to'
                  f'go...')

        template = '[ {sample} {{stage}}/{{total_stages}} {{{{timestamp}}}} ]  '  # noqa: E501
        fkmap_cache_reset()
        for num, (sample, job_grp) in enumerate(jobs_per_sample):
            print(f'{len(jobs) - num} samples to go...')
            abort_sample = False
            # for flag, funcs in script:  # OLD
            stage = 0
            while job_grp:
                job = job_grp.pop(0)
                stage += 1

                t = template.format(sample=sample.sample_id)

                timestamper = Timestamper(
                    template=t.format(stage=stage, total_stages=stage + len(job_grp)),  # noqa: E501
                    file_copy=settings.OMICS_LOADING_LOG,
                )
                with atomic(), timestamper:
                    if stage == 1:
                        print(f'--> {type(job).__name__}')

                    try:
                        job()
                    except KeyboardInterrupt as e:
                        print(repr(e))
                        raise
                    except Exception as e:
                        msg = (f'FAIL: {e.__class__.__name__} "{e}": on '
                               f'{sample.sample_id} at or near {job.run=}')
                        print(msg)
                        # If we're configured to write a log file, then print
                        # the stack to a special FAIL.log file and continue
                        # with the next sample. This optimizes for the case
                        # that the error is caused by occasional unusual data
                        # for individual samples and not a regular bug, which
                        # would trigger on every sample.
                        log = settings.OMICS_LOADING_LOG
                        if not log:
                            raise
                        faillog = Path(log).with_suffix(
                            f'.{sample.sample_id}.FAIL.log'
                        )
                        with faillog.open('w') as ofile:
                            ofile.write(msg + '\n')
                            traceback.print_exc(file=ofile)
                        print(f'see traceback at {faillog}')
                        # skip to next sample, do not set the flag, fn() is
                        # assumed to have rolled back any its own changes to
                        # the DB
                        abort_sample = True
                        break
                    else:
                        # add newly ready jobs
                        for i in reversed(job.before):
                            if i not in job_grp:
                                if i.is_ready(use_cache=False):
                                    job_grp.insert(0, i)

            if abort_sample:
                print(f'Aborting {sample.sample_id}/{sample}!', end=' ')
            else:
                print(f'Sample {sample.sample_id}/{sample} done!', end=' ')
        print()


class TaxonAbundanceLoader(TaxNodeMixin, SampleLoadMixin, BulkLoader):
    """ loader manager for the TaxonAbundance model """
    def process_tpm(self, value, obj):
        # obj.taxon may be None for unclassified or deleted taxids
        # tpm values for those get added together
        taxid = getattr(obj.taxon, 'taxid', None)
        is_duplicate = taxid in self.tpm_data

        # save value for post-processing
        self.tpm_data[taxid] += float(value)

        if is_duplicate:
            # is a row for merged taxid (duplicate)
            raise SkipRow('is duplicate for merged taxid', log=False)
        else:
            # zero as temp placeholder
            return 0.0

    spec = CSV_Spec(
        ('tax_id', 'taxon', 'check_taxid'),
        ('abund_w_subtax', None),
        ('abund_direct', 'tpm', process_tpm),
    )

    def get_file(self, sample):
        """ Get path to lca_abund_summarized file """
        return sample.get_omics_file('TAX_ABUND')

    @atomic_dry
    def load_sample(self, sample, **kwargs):
        """
        Load abundance vs. taxa from lca_abund_summarized files

        The input data may be based on an older version of NCBI's taxonomy.  In
        such a case, abundance for deleted taxids is added to 'unclassified.'
        Merged taxids' abundance is accumulated appropriately.  Abundance for
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
    def get_krona_path(self, sample):
        return Path(settings.KRONA_CACHE_DIR) / f'{sample.sample_id}.html'

    def make_all_krona_charts(self, keep_existing=True):
        """
        Create krona charts in chache directory

        Makes charts for all samples for which there is some tax abundance.
        The default is to keep existing charts.  To re-create charts set
        keep_existing to False.
        """
        Sample = get_sample_model()
        for i in Sample.objects.exclude(taxonabundance=None).only('sample_id'):
            path = self.get_krona_path(i)
            if path.exists():
                if keep_existing:
                    continue
                else:
                    path.unlink()

            saved = self.make_krona_html(i, outpath=path)
            print(f'Saved as: {saved}')

    def as_krona_input_text(self, sample):
        """
        Generate input text data for krona

        Will return an empty string if no abundance data is saved for the given
        sample.
        """
        qs = self.filter(sample=sample).select_related('taxon')
        qs = qs.prefetch_related('taxon__ancestors')
        names = dict(
            TaxName.objects
            .filter(name_class=11, node__abundance__sample=sample)
            .values_list('node_id', 'name')
        )

        rows = []
        for obj in qs:
            row = [str(obj.tpm)]
            if obj.taxon is None:
                row.append('unclassified')
            else:
                for i in obj.taxon.lineage:
                    row.append(names[i.pk])
            rows.append('\t'.join(row))

        if not rows:
            # The krona chart needs some abundance data to work with.  So, no
            # rows should mean no abundance is loaded for the given sample, so
            # a proper course of action is to raise DoesNotExist which the view
            # should catch.  That's why there is a get() call below.  The
            # try/except dance is there to make this all very explicit.
            try:
                qs.get()
            except self.model.DoesNotExist:
                raise
            except Exception as e:
                msg = 'was really expecting DoesNotExist'
                raise RuntimeError(msg) from e

        return '\n'.join(rows)

    def as_krona_html(self, sample, outpath=None, use_cache=True):
        """
        Return krona html chart

        If outpath is given this is save to the file system, if it is None, the
        default, then Krona's html content is the return value (indended to be
        used by a view).  In the latter case a None return value indicates
        either an error with running the krona generator or missing abundance
        data (Check the django error log for details.)
        """
        if use_cache:
            path = self.get_krona_path(sample)
            if not path.exists():
                self.make_krona_html(sample, outpath=path)

            with path.open() as ifile:
                try:
                    return ifile.read()
                except Exception as e:
                    log.error(
                        f'krona html read failed: {e.__class__.__name__}:{e}\n'
                        f'{ifile=}'
                    )
                    ifile.seek(0)
                    return ifile.read(encoding='utf-8')

        else:
            return self.make_krona_html(sample)

    def make_krona_html(self, sample, outpath=None):
        with tempfile.TemporaryDirectory() as tmpd:
            krona_in = tmpd + '/data.txt'
            krona_out = tmpd + '/krona.html'

            with open(krona_in, 'w') as ofile:
                ofile.write(self.as_krona_input_text(sample))

            cmd = [
                'ktImportText',
                '-n', f'all of {sample}',
                '-o', krona_out,
                krona_in
            ]
            subprocess.run(cmd, cwd=tmpd, check=True)

            if outpath is None:
                with open(krona_out) as ifile:
                    return ifile.read()
            else:
                # copy2 returns outpath on success
                return shutil.copy2(krona_out, outpath)


class FileManager(Manager):
    def get_instance(self, sample, filetype, only_new=False):
        """
        Get object, if possible from the DB, but don't save a new object.
        """
        if isinstance(filetype, str):
            filetype = self.model.Type[filetype]

        try:
            obj = self.all().get(sample=sample, filetype=filetype)
        except self.model.DoesNotExist as e:
            if filetype not in self.model.PATH_TAILS:
                raise ValueError(
                    f'File type:{filetype} is not supported by this method (I '
                    f'don\'t know how to compute the path to the file.'
                )

            tail = self.model.PATH_TAILS[filetype]
            if isinstance(tail, str):
                tail = tail.format(sample=sample)

            if not sample.analysis_dir:
                raise RuntimeError(
                    f'can\'t create instance: sample.analysis_dir is blank '
                    f'for {sample}'
                ) from e

            obj = self.model(
                path=settings.OMICS_PIPELINE_DATA / sample.analysis_dir / tail,
                public=None,
                filetype=filetype,
                sample=sample,
            )
        else:
            if only_new:
                raise RuntimeError(f'File object already exists: {obj}')
        return obj


class SampleTrackingManager(Manager):
    @cached_property
    def job_classes(self):
        """
        Make job classes available via flag

        This is a cached property to avoid circular import
        """
        registry = import_string('mibios.omics.tracking.registry')
        classes = {}
        for name, cls in registry.jobs.items():
            classes[cls.flag] = cls
        return classes

    def ready_for_sample(self, sample):
        ready = []
        for obj in self.all().filter(sample=sample):
            for i in obj.job.before:
                if i.is_ready() and i not in ready:
                    ready.append(i)
        return ready
