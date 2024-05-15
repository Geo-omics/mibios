from logging import getLogger

from django.db import models

from mibios import get_registry
from mibios.ncbi_taxonomy.models import TaxNode

from .fields import AccessionField
from . import manager
from .model_utils import (
    ch_opt, fk_opt, fk_req, VocabularyModel, delete_all_objects_quickly,
    Model
)


log = getLogger(__name__)


class CompoundRecord(Model):
    """ Reference DB's entry for chemical compound, reactant, or product """
    DB_BIOCYC = 'BC'
    DB_CHEBI = 'CH'
    DB_HMDB = 'HM'
    DB_PATHBANK = 'PB'
    DB_KEGG = 'KG'
    DB_PUBCHEM = 'PC'
    DB_CHOICES = (
        (DB_BIOCYC, 'Biocyc'),
        (DB_CHEBI, 'ChEBI'),
        (DB_HMDB, 'HMDB'),
        (DB_PATHBANK, 'PathBank'),
        (DB_KEGG, 'KEGG'),
        (DB_PUBCHEM, 'PubChem'),
    )

    accession = models.TextField(max_length=32, unique=True)
    source = models.CharField(max_length=2, choices=DB_CHOICES, db_index=True)
    formula = models.TextField(max_length=64, blank=True)
    charge = models.SmallIntegerField(blank=True, null=True)
    mass = models.TextField(max_length=32, blank=True)  # TODO: decimal??
    names = models.ManyToManyField('CompoundName')
    others = models.ManyToManyField('self', symmetrical=False)

    loader = manager.CompoundRecordLoader()

    def __str__(self):
        return f'{self.get_source_display()}:{self.accession}'

    def group(self):
        """ return QuerySet of synonym/related compound entry group """
        return self.compound.group.all()

    external_urls = {
        DB_BIOCYC: 'https://biocyc.org/compound?orgid=META&id={}',
        DB_CHEBI: 'https://www.ebi.ac.uk/chebi/searchId.do?chebiId={}',
        DB_HMDB: 'https://hmdb.ca/metabolites/{}',
        DB_PATHBANK: None,  # TODO
        DB_KEGG: 'https://www.kegg.jp/entry/{}',
        DB_PUBCHEM: (
            'https://pubchem.ncbi.nlm.nih.gov/compound/{}',
            lambda x: x.removeprefix('CID:')
        ),
    }

    def get_external_url(self):
        url_spec = self.external_urls[self.source]
        if not url_spec:
            return None
        elif isinstance(url_spec, str):
            # assume simple formatting string
            return url_spec.format(self.accession)
        else:
            # assumme a tuple (templ, func)
            return url_spec[0].format(url_spec[1](self.accession))


class CompoundName(VocabularyModel):
    abundance_accessor = 'compoundentry__abundance'


class FunctionName(VocabularyModel):
    abundance_accessor = 'funcrefdbentry__abundance'


class Location(VocabularyModel):
    class Meta(Model.Meta):
        verbose_name = 'subcellular location'


class ReactionCompound(models.Model):
    """
    intermediate model for reaction -> l/r-compound m2m relations

    This is an ordinary django.db.models.Model
    """
    DB_BIOCYC = 'BC'
    DB_KEGG = 'KG'
    DB_RHEA = 'CH'
    SRC_CHOICES = (
        (DB_BIOCYC, 'Biocyc'),
        (DB_KEGG, 'KEGG'),
        (DB_RHEA, 'RHEA'),
    )
    SIDE_CHOICES = ((True, 'left'), (False, 'right'))
    LOCATION_CHOICES = ((True, 'INSIDE'), (False, 'OUTSIDE'))
    TRANSPORT_CHOICES = (
        ('BI', 'BIPORT'),
        ('EX', 'EXPORT'),
        ('IM', 'IMPORT'),
        ('NO', 'NOPORT'),
    )

    # source and target field names correspond to what an automatic through
    # model would have
    reactionrecord = models.ForeignKey('ReactionRecord', **fk_req)
    compoundrecord = models.ForeignKey('CompoundRecord', **fk_req)
    source = models.CharField(max_length=2, choices=SRC_CHOICES, db_index=True)
    side = models.BooleanField(choices=SIDE_CHOICES)
    location = models.BooleanField(choices=LOCATION_CHOICES)
    transport = models.CharField(max_length=2, choices=TRANSPORT_CHOICES)

    class Meta:
        unique_together = (
            ('reactionrecord', 'compoundrecord', 'source', 'side'),
        )

    def __str__(self):
        return (
            f'{self.reactionrecord.accession}<>{self.compoundrecord.accession}'
            f' {self.source} {self.get_side_display()}'
        )


class ReactionRecord(Model):
    SRC_CHOICES = ReactionCompound.SRC_CHOICES
    DIRECT_CHOICES = (
        (True, 'BOTH'),
        (False, 'LTR'),
    )
    accession = AccessionField(max_length=96)
    source = models.CharField(max_length=2, choices=SRC_CHOICES, db_index=True)
    direction = models.BooleanField(
        choices=DIRECT_CHOICES,
        blank=True,
        null=True,
    )
    others = models.ManyToManyField('self', symmetrical=False)
    compound = models.ManyToManyField(
        CompoundRecord,
        through=ReactionCompound,
    )
    uniprot = models.ManyToManyField('Uniprot')
    ec = models.ForeignKey('FuncRefDBEntry', **fk_opt)

    loader = manager.ReactionRecordLoader()

    def __str__(self):
        return self.accession


class FuncRefDBEntry(Model):
    DB_COG = 'cg'
    DB_EC = 'ec'
    DB_GO = 'go'
    DB_IPR = 'ip'
    DB_PFAM = 'pf'
    DB_TCDB = 'tc'
    DB_TIGR = 'ti'
    DB_CHOICES = (
        # by order of UNIREF input file columns
        (DB_TCDB, 'TCDB'),
        (DB_COG, 'COG'),
        (DB_PFAM, 'Pfam'),
        (DB_TIGR, 'TIGR'),
        (DB_GO, 'GO'),
        (DB_IPR, 'InterPro'),
        (DB_EC, 'EC'),
    )
    accession = AccessionField()
    db = models.CharField(max_length=2, choices=DB_CHOICES, db_index=True)
    names = models.ManyToManyField('FunctionName')

    name_loader = manager.FuncRefDBEntryLoader()

    class Meta(Model.Meta):
        verbose_name = 'Function Ref DB Entry'
        verbose_name_plural = 'Func Ref DB Entries'

    def __str__(self):
        return self.accession

    external_urls = {
        DB_COG: 'https://www.ncbi.nlm.nih.gov/research/cog/cog/{}/',
        DB_EC: '',
        DB_GO: 'http://amigo.geneontology.org/amigo/term/{}',
        DB_IPR: 'https://www.ebi.ac.uk/interpro/entry/InterPro/{}/',
        DB_PFAM: 'https://pfam.xfam.org/family/{}',
        DB_TIGR: '',
        DB_TCDB: '',
    }

    def get_external_url(self):
        url_spec = self.external_urls.get(self.db, None)
        if not url_spec:
            return None
        elif isinstance(url_spec, str):
            # assume simple formatting string
            return url_spec.format(self.accession)
        else:
            # assumme a tuple (templ, func)
            return url_spec[0].format(url_spec[1](self.accession))


class Uniprot(Model):
    accession = AccessionField(verbose_name='uniprot id')

    class Meta(Model.Meta):
        verbose_name = 'Uniprot'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    def get_external_url(self):
        return f'https://www.uniprot.org/uniprot/{self.accession}'


class UniRef100(Model):
    """
    Model for UniRef100 clusters
    """
    # The field comments below are based on the columns in
    # OUT_UNIREF.txt in order.

    #  1 UR100
    accession = AccessionField()
    #  2 UR90
    uniref90 = AccessionField(prefix='UNIREF90_', unique=False)
    #  3 Name
    function_names = models.ManyToManyField(FunctionName)
    #  4 Length
    length = models.PositiveIntegerField(blank=True, null=True)
    #  5 SigPep
    signal_peptide = models.TextField(max_length=32, **ch_opt)
    #  6 TMS
    tms = models.TextField(**ch_opt)
    #  7 DNA
    dna_binding = models.TextField(**ch_opt)
    #  8 TaxonId
    taxa = models.ManyToManyField(
        TaxNode,
        related_name='classified_uniref100',
    )
    #  9 Binding
    binding = models.ManyToManyField(CompoundRecord)
    # 10 Loc
    subcellular_locations = models.ManyToManyField(Location)
    # 11-17 TCDB COG Pfam Tigr Gene_Ont InterPro ECs
    function_refs = models.ManyToManyField(FuncRefDBEntry)
    # 18-20 kegg rhea biocyc
    reactions = models.ManyToManyField(ReactionRecord)

    loader = manager.UniRef100Loader()

    class Meta:
        verbose_name = 'UniRef100'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    def get_external_url(self):
        return f'https://www.uniprot.org/uniref/UniRef100_{self.accession}'


# development stuff
def delete_all_uniref100_etc():
    r = get_registry()
    for i in r.apps['mibios.umrad'].get_models():
        if i._meta.model_name.startswith('tax'):
            continue
        print(f'Deleting: {i} ', end='', flush=True)
        delete_all_objects_quickly(i)
        print('[done]')


def load_umrad():
    """ load all of UMRAD from scratch, assuming an empty DB """
    CompoundRecord.loader.load(skip_on_error=True)
    ReactionRecord.loader.load(skip_on_error=True)
    UniRef100.loader.load(skip_on_error=True)
    FuncRefDBEntry.name_loader.load()
