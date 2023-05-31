"""
Module for data load managers

=== Workflow to load metagenomic data ===

Assumes that UMRAD and sample meta-data is loaded
    Sample.objects.update_analysis_status()
    s = Sample.objects.get(sample_id='samp_14')
    Contig.loader.load_fasta_sample(s)
    Gene.loader.load_fasta_sample(s)
    Contig.loader.load_abundance_sample(s)
    Gene.loader.load_abundance_sample(s)
    Alignment.loader.load_sample(s)
    Gene.loader.assign_gene_lca(s)
    Contig.loader.assign_contig_lca(s)
    TaxonAbundance.loader.load_sample(s)
"""

from itertools import groupby, islice
from logging import getLogger
from operator import itemgetter
import os
import shutil
from statistics import median
import subprocess
import tempfile

from django.conf import settings
from django.utils.module_loading import import_string

from mibios.models import QuerySet
from mibios.ncbi_taxonomy.models import TaxNode
from mibios.umrad.models import UniRef100
from mibios.umrad.manager import BulkLoader, Manager
from mibios.umrad.utils import CSV_Spec, atomic_dry

from . import get_sample_model
from .utils import get_fasta_sequence


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


class BBMap_RPKM_Spec(CSV_Spec):
    def process_header(self, file):
        # skip initial non-rows
        pos = file.tell()
        while True:
            line = file.readline()
            if line.startswith('#'):
                if line.startswith('#Name'):
                    file.seek(pos)
                    return super().process_header(file)
                else:
                    # TODO: process rpkm header data
                    pos = file.tell()
            else:
                raise RuntimeError(
                    f'rpkm header lines must start with #, column header with '
                    f'#Name, got: {line}'
                )


class SampleLoadMixin:
    """ Mixin for Loader class that loads per-sample files """

    load_flag_attr = None
    """ may be specified by implementing class """

    sample = None
    """ sample is set by load_sample() for use in per-field helper methods """

    @atomic_dry
    def load_sample(self, sample, **kwargs):
        if 'flag' in kwargs:
            flag = kwargs.pop('flag')
            if flag is None:
                # explicit override / no flag check/set
                pass
        else:
            flag = self.load_flag_attr

        if flag and not kwargs.get('update', False) and getattr(sample, flag):
            raise RuntimeError(f'data already loaded: {flag} -> {sample}')

        if 'file' not in kwargs:
            kwargs.update(file=self.get_file(sample))

        self.sample = sample
        self.load(template={'sample': sample}, **kwargs)
        # ensure subsequent calls of manager methods never get wrong sample:
        self.sample = None

        if flag:
            setattr(sample, flag, True)
            sample.save()


class M8Spec(CSV_Spec):
    def iterrows(self):
        """
        Transform m8 file into top good hits

        Yields rows of triplets [gene, uniref100, score]

        m8 file format:

             0 qseqid  /  $gene
             1 qlen
             2 sseqid  / $hit
             3 slen
             4 qstart
             5 qend
             6 sstart
             7 send
             8 evalue
             9 pident / $pid
            10 mismatch
            11 qcovhsp  /  $cov
            12 scovhsp
        """
        minid = self.loader.MINID
        mincov = self.loader.MINCOV
        score_margin = self.loader.TOP
        per_gene = groupby(super().iterrows(), key=itemgetter(0))
        for gene_id, grp in per_gene:
            # 1. get good hits
            hits = []
            for row in grp:
                pid = float(row[9])
                if pid < minid:
                    continue
                cov = float(row[11])
                if cov < mincov:
                    continue
                # hits: (gene, uniref100, score)
                hits.append((row[0], row[2], int(pid * cov)))

            # 2. keep top hits
            if hits:
                hits = sorted(hits, key=lambda x: -x[2])  # sort by score
                minscore = int(hits[0][2] * score_margin)
                for i in hits:
                    # The int() above are improper rounding, so some scores a
                    # bit below the cutoff will slip through
                    if i[2] >= minscore:
                        yield i


class AlignmentLoader(BulkLoader, SampleLoadMixin):
    """
    loads data from XX_GENES.m8 files into Alignment table

    Parameters:
      1. min query coverage (60%)
      2. min pct alignment identity (60%)
      3. min number of genes to keep (3) (not including? phages/viruses)
      4. top score margin (0.9)
      5. min number of model genomes for func. models (5)

    Notes on Annotatecontigs.pl:
        top == 0.9
        minimum pident == 60%
        minimum cov/qcovhsp == 60
        score = pid*cov
        keep top score per hit (global?)
        keep top score of hits for each gene
        keep top x percentile hits for each gene


    """
    MINID = 60.0
    MINCOV = 60.0
    TOP = 0.9

    load_flag_attr = 'gene_alignment_hits_loaded'

    def get_file(self, sample):
        return sample.get_metagenome_path() / f'{sample.sample_id}_GENES.m8'

    def query2gene_pk(self, value, obj):
        """
        Pre-processor to get gene PK from qseqid column for genes

        The query column has all that's in the fasta header, incl. the prodigal
        data.  Also add the sample pk (gene is a FK field).
        """
        return self.gene_id_map[value]

    def upper(self, value, obj):
        """
        Pre-processor to upper-case the uniref100 id

        the incoming prefixes have mixed case
        """
        return value.upper()

    spec = M8Spec(
        ('gene.pk', query2gene_pk),
        ('ref', upper),
        ('score',),
    )

    def load_sample(self, sample, file=None, **kwargs):
        self.gene_id_map = dict(sample.gene_set.values_list('gene_id', 'pk'))
        if file is None:
            file = self.get_file(sample)
        super().load(file=file, **kwargs)


class CompoundAbundanceLoader(BulkLoader, SampleLoadMixin):
    """ loader manager for CompoundAbundance """
    ok_field_name = 'comp_abund_ok'

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

    Provides the load_fasta_sample() method.
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
        Return extra kwargs for from_sample_fasta()

        Should be overwritten by inheriting class if needed
        """
        return {}

    @atomic_dry
    def load_fasta_sample(self, sample, start=0, limit=None, bulk=True,
                          validate=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        if getattr(sample, self.fasta_load_flag):
            raise RuntimeError(
                f'data already loaded - update not supported: '
                f'{self.fasta_load_flag} -> {sample}'
            )

        objs = self.from_sample_fasta(sample, start=start, limit=limit)
        if validate:
            objs = ((i for i in objs if i.full_clean() or True))

        if bulk:
            self.bulk_create(objs)
        else:
            for i in objs:
                i.save()

        setattr(sample, self.fasta_load_flag, True)
        sample.save()

    def from_sample_fasta(self, sample, start=0, limit=None):
        """
        Generate instances for given sample

        Helper for load_fasta_sample().
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


class ContigLikeLoader(SequenceLikeLoader):
    """ Manager for ContigLike abstract model """
    abundance_load_flag = None  # set in inheriting class

    def get_contigs_file_path(self, sample):
        # FIXME: this is unused, remove?
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.sample_id}_contigs_*.txt'
        )

    def process_coverage_header_data(self, sample, data):
        """ Add header data to sample """
        # must be implemented by inheriting class
        raise NotImplementedError

    rpkm_spec = None
    """ rpkm_spec must be set in inheriting classes """

    @atomic_dry
    def load_abundance_sample(self, sample, file=None, **kwargs):
        """
        Load data from bbmap *.rpkm files

        Assumes that fasta data was loaded previously.
        """
        if file is None:
            file = self.get_rpkm_path(sample)

        # read header data
        reads = None
        mapped = None
        with open(file) as ifile:
            print(f'Reading from {ifile.name} ...')
            while True:
                pos = ifile.tell()
                line = ifile.readline()
                if line.startswith('#Name\t') or not line.startswith('#'):
                    # we're past the preamble
                    ifile.seek(pos)
                    break
                key, _, data = line.strip().partition('\t')
                if key == '#Reads':
                    reads = int(data)
                elif key == '#Mapped':
                    mapped = int(data)

        if reads is None or mapped is None:
            raise RuntimeError('Failed parsing (mapped) read counts')

        if sample.read_count is not None and sample.read_count != reads:
            print(f'Warning: overwriting existing read count with'
                  f'different value: {sample.read_count}->{reads}')
        sample.read_count = reads

        mapped_old_val = getattr(sample, self.reads_mapped_sample_attr)
        if mapped_old_val is not None and mapped_old_val != mapped:
            print(f'Warning: overwriting existing mapped read count with'
                  f'different value: {mapped_old_val}->{mapped}')
        setattr(sample, self.reads_mapped_sample_attr, mapped)
        sample.save()

        self.load_sample(
            sample,
            flag=self.abundance_load_flag,
            spec=self.rpkm_spec,
            file=file,
            update=True,
            **kwargs)

    @staticmethod
    def get_lca(nodes):
        """ helper to calculate the LCA of given TaxNodes """
        lca, *others = nodes
        for i in others:
            lca = lca.lca(i)
        lins = []
        for i in nodes:
            # assume ancestors are prefetched, so sort by python
            lineage = sorted(i.ancestors.all(), key=lambda x: x.rank)
            lineage.append(i)
            lins.append(lineage)

        lca = None
        for items in zip(*lins):
            if len(set((i.pk for i in items))) == 1:
                lca = items[0]
            else:
                break

        return lca

    def abundance_stats(self, sample):
        """
        Calculate various abundance based statistics per taxon and sample

        Return value is intended to be fed into the per-taxon abundance table.
        """
        qs = (self
              .filter(sample=sample)
              .exclude(lca=None)
              .select_related('lca')
              .order_by('-lca__rank', 'lca'))
        stats = {}
        for lca, objs in groupby(qs.iterator(), key=lambda x: x.lca):
            objs = list(objs)

            # weighted fpkm averages
            wmedian_fpkm = median((i.fpkm for i in objs for _ in range(i.length)))  # noqa:E501

            stats[lca] = {}
            stats[lca]['wmedian_fpkm'] = wmedian_fpkm
            stats[lca]['count'] = len(objs)
            stats[lca]['length'] = sum((i.length for i in objs))
            stats[lca]['reads_mapped'] = sum((i.reads_mapped for i in objs))
            stats[lca]['frags_mapped'] = sum((i.frags_mapped for i in objs))

        return stats


class ContigLoader(ContigLikeLoader):
    """ Manager for the Contig model """
    fasta_load_flag = 'contig_fasta_loaded'
    abundance_load_flag = 'contigs_abundance_loaded'
    reads_mapped_sample_attr = 'reads_mapped_contigs'

    def get_fasta_path(self, sample):
        return sample.get_metagenome_path() / 'assembly' \
            / 'megahit_noNORM' / 'final.contigs.renamed.fa'

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'assembly' \
            / f'{sample.sample_id}_READSvsCONTIGS.rpkm'

    def trim_id(self, value, obj):
        """ Pre-processor to trim tracking id off, e.g. deadbeef_123 => 123 """
        _, _, value = value.partition('_')
        return value.upper()

    def calc_rpkm(self, value, obj):
        """ calculate rpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.reads_mapped)
                / int(obj.length) / self.sample.read_count)

    def calc_fpkm(self, value, obj):
        """ calculate fpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.frags_mapped)
                / int(obj.length) / self.sample.read_count)

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'contig_id', trim_id),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm_bbmap'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm_bbmap'),
        (BBMap_RPKM_Spec.CALC_VALUE, 'rpkm', calc_rpkm),
        (BBMap_RPKM_Spec.CALC_VALUE, 'fpkm', calc_fpkm),
    )

    @atomic_dry
    def assign_contig_lca(self, sample):
        """
        assign / pre-compute taxa and LCAs to contigs via genes

        This populates the Contig.lca/taxa fields
        """
        Gene = import_string('mibios.omics.models.Gene')
        genes = Gene.objects.filter(sample=sample)
        # genes = genes.exclude(hits=None)
        genes = genes.values_list('contig_id', 'pk')
        genes = genes.order_by('contig_id')
        print('Fetching taxonomy... ', end='', flush=True)
        taxa = TaxNode.objects \
            .filter(classified_gene__sample=sample) \
            .distinct() \
            .in_bulk()
        print(f'{len(taxa)} [OK]')
        print('Fetching contigs... ', end='', flush=True)
        contigs = self.model.objects.filter(sample=sample).in_bulk()
        print(f'{len(contigs)} [OK]')
        print('Fetching Gene -> TaxNode links... ', end='', flush=True)
        g2tax_qs = Gene._meta.get_field('taxa').remote_field.through.objects
        g2tax_qs = g2tax_qs.filter(gene__sample=sample)
        g2tax_qs = g2tax_qs.values_list('gene_id', 'taxnode_id')
        g2tax_qs = groupby(g2tax_qs.order_by('gene_id'), key=lambda x: x[0])
        g2taxa = {}
        for gene_pk, grp in g2tax_qs:
            g2taxa[gene_pk] = [i for _, i in grp]
        print(f'{len(g2taxa)} [OK]')

        lca_cache = {}

        def contig_taxnode_links():
            """ generate m2m links with side-effects """
            for contig_pk, grp in groupby(genes, lambda x: x[0]):
                taxnode_pks = set()
                for _, gene_pk, in grp:
                    try:
                        taxnode_pks.update(g2taxa[gene_pk])
                    except KeyError:
                        # gene w/o hits
                        pass

                # assign contig LCA from gene LCAs (side-effect):
                if taxnode_pks:
                    taxnode_pks = frozenset(taxnode_pks)
                    try:
                        lca = lca_cache[taxnode_pks]
                    except KeyError:
                        node, *others = ((taxa[i] for i in taxnode_pks))
                        lca_lin = node.get_lca_lineage(*others, full=True)
                        if lca_lin:
                            lca = lca_lin[-1]
                        else:
                            lca = None
                        lca_cache[taxnode_pks] = lca
                else:
                    lca = None

                contigs[contig_pk].lca = lca

                for i in taxnode_pks:
                    yield (contig_pk, i)

        Through = self.model._meta.get_field('taxa').remote_field.through
        delcount, _ = Through.objects.filter(contig__sample=sample).delete()
        if delcount:
            print(f'Deleted {delcount} existing contig-taxnode links')
        objs = (
            Through(contig_id=i, taxnode_id=j)
            for i, j in contig_taxnode_links()
        )
        self.bulk_create_wrapper(Through.objects.bulk_create)(objs)
        if True:  # flake8 trickery
            del lca_cache

        self.fast_bulk_update(contigs.values(), fields=['lca'])


class FuncAbundanceLoader(BulkLoader, SampleLoadMixin):
    ok_field_name = 'func_abund_ok'

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.sample_id}_functionss_*.txt'
        )

    spec = CSV_Spec(
        ('fid', 'function'),
        ('fname', None),
        ('FUNC_GENES', None),
        ('FUNC_SCOS', 'scos'),
        ('FUNC_RPKM', 'rpkm'),
        ('fidlca', None),
    )


class GeneLoader(ContigLikeLoader):
    """ Manager for the Gene model """
    fasta_load_flag = 'gene_fasta_loaded'
    abundance_load_flag = 'contigs_abundance_loaded'
    reads_mapped_sample_attr = 'reads_mapped_genes'

    def get_fasta_path(self, sample):
        return (
            sample.get_metagenome_path() / 'genes'
            / f'{sample.sample_id}_GENES.fna'
        )

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'genes' \
            / f'{sample.sample_id}_READSvsGENES.rpkm'

    def extract_gene_id(self, value, obj):
        """
        Pre-processor to get just the gene id from prodigal fasta header
        """
        # deadbeef_123_1 # bla bla bla => 123_1
        return value.split(maxsplit=1)[0].partition('_')[2]

    def calc_rpkm(self, value, obj):
        """ calculate rpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.reads_mapped)
                / int(obj.length) / self.sample.read_count)

    def calc_fpkm(self, value, obj):
        """ calculate fpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.frags_mapped)
                / int(obj.length) / self.sample.read_count)

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'gene_id', extract_gene_id),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm_bbmap'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm_bbmap'),
        (BBMap_RPKM_Spec.CALC_VALUE, 'rpkm', calc_rpkm),
        (BBMap_RPKM_Spec.CALC_VALUE, 'fpkm', calc_fpkm),
    )

    def get_set_from_fa_head_extra_kw(self, sample):
        # returns a dict of the sample's contigs
        qs = sample.contig_set.values_list('contig_id', 'pk')
        return dict(contig_id_map=dict(qs.iterator()))

    @atomic_dry
    def assign_gene_lca(self, sample):
        """
        assign / pre-compute taxa and LCAs to genes via uniref100 hits

        This also populates Gene.lca / Gene.besthit fields
        """
        Alignment = import_string('mibios.omics.models.Alignment')

        print('Fetching uniref100/taxa...', end='', flush=True)
        qs = UniRef100.objects \
            .filter(gene_hit__sample=sample) \
            .prefetch_related('taxa') \
            .distinct()
        u2t = {i.pk: i.taxa.all() for i in qs.iterator()}
        print(f'{len(u2t)} [OK]')

        hits = Alignment.objects.filter(gene__sample=sample)
        hits = hits.select_related('gene').order_by('gene')
        lca_cache = {}
        genes = []

        def gene_taxa_links():
            """
            generate pk pairs (gene, taxnode) to make Gene<->TaxNode m2m links

            As a side-effect this also sets besthit and lca for each gene that
            has a hit.
            """
            for gene, grp in groupby(hits.iterator(), key=lambda x: x.gene):
                # get all taxa for all hits, some UR100 may not have taxa
                taxa = {}
                best = None
                for align in grp:
                    if best is None or align.score > best.score:
                        best = align
                    for taxnode in u2t.get(align.ref_id, []):
                        if taxnode.pk in taxa:
                            continue
                        else:
                            taxa[taxnode.pk] = taxnode
                            yield (gene.pk, taxnode.pk)

                gene.besthit_id = best.ref_id
                if taxa:
                    node_pks = frozenset(taxa.keys())
                    try:
                        lca = lca_cache[node_pks]
                    except KeyError:
                        node, *others = taxa.values()
                        lca = node.get_lca_lineage(*others, full=True)
                        lca_cache[node_pks] = lca
                    if lca:
                        gene.lca = lca[-1]
                    else:
                        # LCA is root
                        gene.lca = None
                else:
                    # some genes here have hits but no taxa, so no LCA (=root)
                    gene.lca = None
                genes.append(gene)

        Through = self.model._meta.get_field('taxa').remote_field.through
        delcount, _ = Through.objects.filter(gene__sample=sample).delete()
        if delcount:
            print(f'Deleted {delcount} existing gene-taxnode links')
        objs = (Through(gene_id=i, taxnode_id=j) for i, j in gene_taxa_links())
        self.bulk_create_wrapper(Through.objects.bulk_create)(objs)
        if True:  # tricking flake8, undefined name in inner function above
            del lca_cache, u2t

        # update lca field for all genes incl. the unknowns
        self.fast_bulk_update(genes, fields=['lca', 'besthit'])
        print('Erasing lca for genes without any hits... ', end='', flush=True)
        num_unknown = self.filter(hits=None).update(lca=None)
        print(f'{num_unknown} [OK]')


class SampleManager(Manager):
    """ Manager for the Sample """
    def get_file(self):
        """ get the metagenomic pipeline import log """
        return settings.OMICS_DATA_ROOT / 'data' / 'imported_samples.tsv'

    @atomic_dry
    def update_analysis_status(self, source_file=None, skip_on_error=False):
        """
        Update sample table with analysis status
        """
        if source_file is None:
            source_file = self.get_file()

        with open(source_file) as f:
            # check assumptions on columns
            SAMPLE_ID = 0
            STUDY_ID = 1
            TYPE = 3
            ANALYSIS_DIR = 4
            SUCCESS = 6
            cols = (
                (SAMPLE_ID, 'SampleID'),
                (STUDY_ID, 'StudyID'),
                (TYPE, 'sample_type'),
                (ANALYSIS_DIR, 'sample_dir'),
                (SUCCESS, 'import_success'),
            )

            head = f.readline().rstrip('\n').split('\t')
            for index, column in cols:
                if head[index] != column:
                    raise RuntimeError(
                        f'unexpected header in {f.name}: 0-based column '
                        f'{index} is {head[index]} but {column} is expected'
                    )

            good_seen = []
            samp_id_seen = set()
            changed = 0
            unchanged = 0
            notfound = 0
            nosuccess = 0
            for line in f:
                row = line.rstrip('\n').split('\t')
                sample_id = row[SAMPLE_ID]
                dataset = row[STUDY_ID]
                sample_type = row[TYPE]
                analysis_dir = row[ANALYSIS_DIR]
                success = row[SUCCESS]

                if not all([sample_id, dataset, sample_type, analysis_dir, success]):  # noqa: E501
                    raise RuntimeError(f'field is empty in row: {row}')

                if sample_id in samp_id_seen:
                    # skip rev line
                    continue
                else:
                    samp_id_seen.add(sample_id)

                if success != 'TRUE':
                    log.info(f'ignoring {sample_id}: no import success')
                    nosuccess += 1
                    continue

                need_save = False
                try:
                    obj = self.get(
                        sample_id=sample_id,
                    )
                except self.model.DoesNotExist:
                    # object may be hidden by default manager (private or so)
                    log.warning(f'no such sample: {sample_id} (skipping)')
                    notfound += 1
                    continue
                else:
                    if obj.dataset.dataset_id != dataset:
                        raise RuntimeError('{sample_id}: bad dataset')
                    if obj.sample_type != sample_type:
                        raise RuntimeError('{sample_id}: bad sample type')

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
                        unchanged += 1

                if need_save:
                    obj.metag_pipeline_reg = True
                    obj.full_clean()
                    obj.save()
                    log.info(save_info)
                    changed += 1

                good_seen.append(obj.pk)

        log.info(f'Summary:\n  samples listed: {len(samp_id_seen)}\n  '
                 f'samples updated: {changed}')
        if notfound:
            log.warning(
                f'Samples missing from database (or hidden): {notfound}'
            )

        if nosuccess:
            log.warning(f'Samples not marked "import_success": {nosuccess}')

        if unchanged:
            log.info(f'Data for {unchanged} samples are already in the DB and '
                     f'remain unchanged')

        missing_or_bad = self.exclude(pk__in=good_seen)
        if missing_or_bad.exists():
            log.warning(f'The DB has {missing_or_bad.count()} samples for '
                        f'which are missing from {source_file} or which had '
                        f'to be skipped')

        missing = self.exclude(sample_id__in=samp_id_seen)
        if missing.exists():
            log.warning(f'The DB has {missing.count()} samples not at all '
                        f'listed in {source_file}')

    def status(self):
        if not self.exists():
            print('No samples in database yet')
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


class TaxonAbundanceManager(Manager):
    """ loader manager for the TaxonAbundance model """
    ok_field_name = 'tax_abund_ok'

    @atomic_dry
    def populate_sample(self, sample, validate=False):
        """
        populate the taxon abundance table for a single sample

        This requires that the sample's genes' LCAs have been set.
        """
        qs = self.filter(sample=sample)
        if qs.exists():
            print('Deleting existing data... ', end='', flush=True)
            _, num_deleted = qs.delete()
            print(f'{num_deleted} [OK]')

        Contig = import_string('mibios.omics.models.Contig')
        Gene = import_string('mibios.omics.models.Gene')
        print('Compiling stats for contigs... ', end='', flush=True)
        contig_stats = Contig.loader.abundance_stats(sample)
        print(f'{len(contig_stats)} [OK]')
        print('Compiling stats for genes... ', end='', flush=True)
        gene_stats = Gene.loader.abundance_stats(sample)
        print(f'{len(gene_stats)} [OK]')

        # taxa for which we have stats don't quite overlap genes vs. contigs,
        # but we'll create an object for each taxon and fill in default values
        # (0.0) for each statistics that is missing
        taxa = set(contig_stats.keys()).union(gene_stats.keys())

        def fpkm(count, length):
            """
            calculate fpkm: count per kb length per 1m sequenced readpairs
            """
            if not count and not length:
                # return 0.0 fpkm for missing data and avoid zero division
                # but still catch bad data (zero length but non-zero count)
                return 0.0
            return 1_000_000_000 * count / (length * sample.read_count)

        total = sample.read_count
        objs = []
        empty_dict = {}  # stand-in so getting the default works
        for taxon in taxa:
            cont_st = contig_stats.get(taxon, empty_dict)
            gene_st = gene_stats.get(taxon, empty_dict)

            len_contigs = cont_st.get('length', 0)
            len_genes = gene_st.get('length', 0)
            frags_mapped_contig = cont_st.get('frags_mapped', 0)
            frags_mapped_gene = gene_st.get('frags_mapped', 0)

            objs.append(self.model(
                sample=sample,
                taxon=taxon,
                count_contig=cont_st.get('count', 0),
                count_gene=gene_st.get('count', 0),
                len_contig=len_contigs,
                len_gene=len_genes,
                mean_fpkm_contig=fpkm(frags_mapped_contig, len_contigs),
                mean_fpkm_gene=fpkm(frags_mapped_gene, len_genes),
                wmedian_fpkm_contig=cont_st.get('wmedian_fpkm', 0.0),
                wmedian_fpkm_gene=gene_st.get('wmedian_fpkm', 0.0),
                norm_reads_contig=cont_st.get('reads_mapped', 0.0) / total,
                norm_reads_gene=gene_st.get('reads_mapped', 0.0) / total,
                norm_frags_contig=frags_mapped_contig / total,
                norm_frags_gene=frags_mapped_gene / total,
            ))
        self.bulk_create(objs)

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
