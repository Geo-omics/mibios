from itertools import zip_longest

from django.db import models

from mibios.umrad.model_utils import ch_opt, fk_opt, fk_req, opt
from mibios.umrad.model_utils import Model as UmradModel
from .managers import CitationLoader, Loader, TaxNodeLoader


class Model(UmradModel):
    loader = Loader()

    class Meta:
        abstract = True


class Citation(Model):
    cit_id = models.PositiveIntegerField(
        unique=True,
        help_text='the unique id of citation',
    )
    cit_key = models.TextField(
        help_text='citation key',
    )
    medline_id = models.PositiveIntegerField(
        **opt,
        help_text='unique id in MedLine database',
    )
    pubmed_id = models.PositiveIntegerField(
        **opt,
        help_text='unique id in PubMed database',
    )
    url = models.TextField(
        help_text='URL associated with citation',
    )
    text = models.TextField(
        help_text='any text (usually article name and authors)'
        """
            The following characters are escaped in this text by a backslash
            newline (appear as "\n"),
            tab character ("\t"),
            double quotes ('\"'),
            backslash character ("\\").
        """
    )
    node = models.ManyToManyField('TaxNode')

    # 0 means blank for these:
    medline_id.empty_values = medline_id.empty_values + [0]
    pubmed_id.empty_values = pubmed_id.empty_values + [0]

    loader = CitationLoader()

    def __str__(self):
        s = f'{self.cit_key[:30]}'
        if self.pubmed_id:
            s += f' (PM:{self.pubmed_id})'
        return s


class DeletedNode(Model):
    taxid = models.PositiveIntegerField(
        unique=True,
        help_text='deleted node id',
    )

    def __str__(self):
        return str(self.taxid)


class Division(Model):
    division_id = models.PositiveIntegerField(
        help_text='taxonomy database division id',
        unique=True,
    )
    cde = models.CharField(
        max_length=3,
        help_text='GenBank division code (three characters)',
    )
    name = models.TextField(unique=True)
    comments = models.TextField(**ch_opt)

    def __str__(self):
        return str(self.name)


class Gencode(Model):
    genetic_code_id = models.PositiveIntegerField(
        help_text='GenBank genetic code id',
        unique=True,
    )
    abbreviation = models.CharField(
        max_length=10,
        **ch_opt,
        help_text='genetic code name abbreviation',
    )
    name = models.TextField(
        unique=True,
        help_text='genetic code name',
    )
    cde = models.TextField(
        **ch_opt,
        help_text='translation table for this genetic code',
    )
    starts = models.TextField(
        **ch_opt,
        help_text='start codons for this genetic code',
    )


class Host(Model):
    node = models.ForeignKey('TaxNode', **fk_req)  # 1-1?
    potential_hosts = models.TextField(
        # FIXME: maybe make this m2m?
        help_text="theoretical host list separated by comma ','",
    )

    class Meta:
        unique_together = (('node', 'potential_hosts'),)

    def __str__(self):
        return f'{self.node}/{self.potential_hosts}'


class MergedNodes(Model):
    old_taxid = models.PositiveIntegerField(
        unique=True,
        help_text='id of nodes which has been merged',
    )
    new_node = models.ForeignKey(
        'TaxNode',
        **fk_req,
        help_text='node which is result of merging',
    )

    def __str__(self):
        return f'{self.old_taxid}->{self.new_node}'


class TaxName(Model):
    NAME_CLASS_SCI = 11
    NAME_CLASSES = (
        (1, 'acronym'),
        (2, 'authority'),
        (3, 'blast name'),
        (4, 'common name'),
        (5, 'equivalent name'),
        (6, 'genbank acronym'),
        (7, 'genbank common name'),
        (8, 'genbank synonym'),
        (9, 'in-part'),
        (10, 'includes'),
        (NAME_CLASS_SCI, 'scientific name'),
        (12, 'synonym'),
    )

    node = models.ForeignKey(
        'TaxNode',
        **fk_req,
        help_text='the node associated with this name',
    )
    name = models.TextField(help_text='the name itself')
    unique_name = models.TextField(
        help_text='the unique variant of this name if name not unique',
    )
    name_class = models.PositiveSmallIntegerField(
        choices=NAME_CLASSES,
        help_text='synonym, common name, ...',
    )

    class Meta:
        unique_together = (
            # don't need the unique_name? Ha!
            ('node', 'name', 'name_class'),
        )
        indexes = [
            models.Index(fields=['node', 'name_class']),
        ]

    def __str__(self):
        return self.name


class TaxNode(Model):
    taxid = models.PositiveIntegerField(
        unique=True,
        verbose_name='taxonomy ID',
    )
    parent = models.ForeignKey('self', **fk_opt, related_name='children')
    rank = models.CharField(max_length=32, db_index=True)
    embl_code = models.CharField(max_length=2, **ch_opt)
    division = models.ForeignKey(Division, **fk_req)
    is_div_inherited = models.BooleanField()
    gencode = models.ForeignKey(
        Gencode,
        **fk_req,
        related_name='node',
    )
    is_gencode_inherited = models.BooleanField()
    mito_gencode = models.ForeignKey(
        Gencode,
        **fk_req,
        related_name='node_mito',
    )
    is_mgc_inherited = models.BooleanField(
        help_text='node inherits mitochondrial gencode from parent',
    )
    is_genbank_hidden = models.BooleanField(
        help_text='name is suppressed in GenBank entry lineage',
    )
    hidden_subtree_root = models.BooleanField(
        help_text='this subtree has no sequence data yet',
    )
    comments = models.TextField(**opt)
    plastid_gencode = models.ForeignKey(
        Gencode,
        **fk_opt,
        related_name='node_plastid',
    )
    is_pgc_inherited = models.BooleanField(
        **opt,
        help_text='node inherits plastid gencode from parent',
    )
    has_specified_species = models.BooleanField(
        help_text='species in the node\'s lineage has formal name'
    )
    hydro_gencode = models.ForeignKey(
        Gencode,
        **fk_opt,  # missing in single row
        related_name='node_hydro',
    )
    is_hgc_inherited = models.BooleanField(
        help_text='inherits hydrogenosome gencode from parent'
    )

    loader = TaxNodeLoader()

    class Meta:
        verbose_name = 'NCBI taxon'
        verbose_name_plural = 'NCBI taxa'

    def __str__(self):
        return f'{self.taxid}'

    @property
    def name(self):
        """
        Get the scientific name of node

        This works because (and as long as) each node has exactly one
        scientific name
        """
        return self.taxname_set.get(name_class=TaxName.NAME_CLASS_SCI)

    def is_root(self):
        """ Say if node is the root of the taxonomic tree """
        return self.taxid == 1

    STANDARD_RANKS = (
        'superkingdom', 'phylum', 'class', 'order', 'family', 'genus',
        'species',
    )

    def lineage_list(self, full=True, names=True):
        lineage = list(reversed(list(self.ancestors(all_ranks=full))))
        if names:
            f = dict(node_id__in=lineage, name_class=TaxName.NAME_CLASS_SCI)
            qs = TaxName.objects.filter(**f)
            name_map = dict(qs.values_list('node__taxid', 'name'))
            lineage = [(i, name_map[i.taxid]) for i in lineage]

        return lineage

    def lineage(self, full=True):
        lin = [
            f'{i.rank}:{name}'
            for i, name in self.lineage_list(full=full, names=True)
        ]
        return ';'.join(lin)

    def ancestors(self, all_ranks=True):
        """
        Generate a node's ancestor nodes starting with itself, towards the root

        Will always yield self even if all_ranks is False.
        """
        cur = self
        while True:
            if all_ranks or cur.rank in self.STANDARD_RANKS or cur is self:
                yield cur
            if cur.is_root():
                break
            cur = cur.parent

    def lca(self, other, all_ranks=True):
        seen_a = set()
        seen_b = set()
        a_anc = self.ancestors(all_ranks=all_ranks)
        b_anc = other.ancestors(all_ranks=all_ranks)
        for a, b in zip_longest(a_anc, b_anc):
            if a is not None:
                seen_a.add(a)
            if b is not None:
                seen_b.add(b)

            if a in seen_b:
                return a
            if b in seen_a:
                return b

        # all_ranks == False but lca is above superkingdom
        return None

    def __eq__(self, other):
        # It's taxid or nothing
        return self.taxid == other.taxid

    def is_ancestor_of(self, other):
        """
        Check if self is ancestor of other

        This method is the base of the rich comparison method
        """
        # TODO: optimize by checking on rank?
        for i in other.ancestors(all_ranks=True):
            if self == i:
                return True
        return False

    def __lt__(self, other):
        return self.is_ancestor_of(other) and not self == other

    def __le__(self, other):
        return self.is_ancestor_of(other)

    def __gt__(self, other):
        return other.is_ancestor_of(self) and not self == other

    def __ge__(self, other):
        return other.is_ancestor_of(self)

    @classmethod
    def search(cls, query, any_name_class=False):
        """ convenience method to get nodes from a (partial) name """
        f = {}
        if not any_name_class:
            f['taxname__name_class'] = TaxName.NAME_CLASS_SCI
        if query[0].isupper():
            f['taxname__name__startswith'] = query
        else:
            f['taxname__name__icontains'] = query
        return cls.objects.filter(**f)  # FIXME: distinct()?


class TypeMaterial(Model):
    # It is not clear if this table has unique or unique-together columns.  The
    # identifier column has entries that only differ by punctuation.
    node = models.ForeignKey(TaxNode, **fk_req)
    tax_name = models.TextField(
        help_text='organism name type material is assigned to',
    )
    material_type = models.ForeignKey(
        'TypeMaterialType',
        **fk_req,
    )
    identifier = models.CharField(
        max_length=32,
        help_text='identifier in type material collection',
    )

    def __str__(self):
        return f'{self.node}/{self.material_type}'


class TypeMaterialType(Model):
    name = models.CharField(
        max_length=64,
        unique=True,
        help_text='name of type material type',
    )
    synonyms = models.CharField(
        max_length=128,
        help_text='alternative names for type material type',
    )
    nomenclature = models.CharField(
        max_length=2,
        help_text='Taxonomic Code of Nomenclature coded by a single letter',
        # B - International Code for algae, fungi and plants (ICN), previously
        # Botanical Code, P - International Code of Nomenclature of Prokaryotes
        # (ICNP), Z - International Code of Zoological Nomenclature (ICZN), V -
        # International Committee on Taxonomy of Viruses (ICTV) virus
        # classification.
    )
    description = models.TextField(
        help_text='descriptive text',
    )

    def __str__(self):
        return f'{self.name}'
