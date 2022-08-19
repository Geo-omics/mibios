"""
GLAMR-specific modeling
"""
from django.db import models, router, transaction
from django.urls import reverse

from mibios_omics.models import AbstractDataset, AbstractSample
from mibios_umrad.fields import AccessionField
from mibios_umrad.models import Model
from mibios_umrad.model_utils import ch_opt, fk_opt, uniq_opt, opt

from .load import DatasetLoader, ReferenceLoader, SampleLoader


class Dataset(AbstractDataset):
    """
    A collection of related samples, e.g. a study or project
    """
    # FIXME: it's not clear which field identifies a "data set", which field
    # may not be blank, and which shall be unique
    study_id = models.PositiveIntegerField(
        unique=True,
        help_text='GLAMR accession to data set/study/project',
    )
    reference = models.ForeignKey('Reference', **fk_opt)
    bioproject = AccessionField(**uniq_opt)
    jgi_project = AccessionField(**uniq_opt)
    gold_id = AccessionField(**uniq_opt)
    scheme = models.CharField(
        max_length=512,
        **ch_opt,
        verbose_name='location and sampling scheme',
    )
    material_type = models.CharField(
        max_length=128,
        **ch_opt,
    )
    water_bodies = models.CharField(
        max_length=256,
        **ch_opt,
        help_text='list or description of sampled bodies of water',
    )
    primers = models.CharField(
        max_length=64,
        **ch_opt,
    )
    sequencing_target = models.CharField(
        max_length=64,
        **ch_opt,
    )
    sequencing_platform = models.CharField(
        max_length=64,
        **ch_opt,
    )
    size_fraction = models.CharField(
        max_length=32,
        **ch_opt,
        help_text='e.g.: >0.22µm or 0.22-1.6µm',
    )
    note = models.TextField(**ch_opt)

    loader = DatasetLoader()
    orphan_group_description = 'samples without a data set'

    class Meta:
        default_manager_name = 'objects'

    def __init__(self, *args, orphan_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_group = orphan_group
        if orphan_group and not self.short_name:
            self.short_name = self.orphan_group_description

    def __str__(self):
        if self.reference_id is None:
            ref = ''
        else:
            ref = self.reference.short_reference
        maxlen = 60 - len(ref)  # max length available for scheme part
        scheme = self.scheme
        if scheme and len(scheme) > maxlen:
            scheme = scheme[:maxlen]
            # remove last word and add [...]
            scheme = ' '.join(scheme.split(' ')[:-1]) + '[\u2026]'

        return ' - '.join(filter(None, [scheme, ref])) or self.short_name \
            or super().__str__()

    bioproject_url_templ = 'https://www.ncbi.nlm.nih.gov/bioproject/{}'
    jgi_project_url_templ = 'https://genome.jgi.doe.gov/portal/lookup' \
        '?keyName=jgiProjectId&keyValue={}&app=Info&showParent=false'
    gold_id_url_templ = None

    def external_urls(self):
        """ collect all external accessions with URLs """
        urls = []
        for i in ['bioproject', 'jgi_project', 'gold_id']:
            field_value = getattr(self, i)
            if field_value is None:
                items = []
            else:
                items = field_value.replace(',', ' ').split()
            for j in items:
                template = getattr(self, i + '_url_templ')
                if i == 'bioproject' and not j.startswith('PRJ'):
                    # TODO: bad accession, fix at source
                    template = None
                if template:
                    urls.append((j, template.format(j)))
                else:
                    urls.append((j, None))
        return urls

    def get_accession_url(self):
        if self.accession_db == self.DB_NCBI:
            return f'https://www.ncbi.nlm.nih.gov/search/all/?term={self.accession}'  # noqa: E501
        elif self.accession_db == self.DB_JGI:
            return f'https://genome.jgi.doe.gov/portal/?core=genome&query={self.accession}'  # noqa: E501

    def get_absolute_url(self):
        if self.orphan_group:
            return reverse('dataset', args=[0])
        return reverse('dataset', args=[self.pk])

    def get_samples_url(self):
        pk = 0 if self.orphan_group else self.pk
        return reverse('dataset_sample_list', args=[pk])


# Create your models here.
class Reference(Model):
    """
    A journal article or similar, primary reference for a data set
    """
    short_reference = models.CharField(
        max_length=128,
        blank=False,
        help_text='short reference',
    )
    authors = models.CharField(
        max_length=2048,
        blank=True,
        help_text='author listing',
    )
    title = models.CharField(
        max_length=512,
        blank=True,
    )
    abstract = models.TextField(blank=True)
    key_words = models.CharField(max_length=128, blank=True)
    publication = models.CharField(max_length=128)
    doi = models.URLField()

    loader = ReferenceLoader()

    class Meta:
        unique_together = (
            # FIXME: this, for now, needs fix in source data
            ('short_reference', 'publication'),
        )

    def __str__(self):
        maxlen = 60
        value = f'{self.short_reference} "{self.title}"'
        if len(value) > maxlen:
            value = value[:maxlen]
            # rm last word, add ellipsis
            value = ' '.join(value.split(' ')[:-1]) + '[\u2026]"'
        return value

    def get_absolute_url(self):
        return reverse('reference', args=[self.pk])


class Sample(AbstractSample):

    # TODO: SampleID ??
    sample_name = models.CharField(max_length=64)
    # TODO: belongs to rdataset? NCBI_BioProject
    biosample = models.CharField(max_length=16, **ch_opt)
    sra_accession = models.CharField(max_length=16, **ch_opt, help_text='SRA accession')  # noqa: E501
    amplicon_target = models.CharField(max_length=16, **ch_opt)
    fwd_primer = models.CharField(max_length=32, **ch_opt)
    rev_primer = models.CharField(max_length=32, **ch_opt)
    geo_loc_name = models.CharField(max_length=256, **ch_opt)
    gaz_id = models.CharField(max_length=16, **ch_opt, verbose_name='GAZ id')
    latitude = models.CharField(max_length=16, **ch_opt)
    longitude = models.CharField(max_length=16, **ch_opt)
    # timestamp: expect ISO8601 formats plus yyyy and yyyy-mm
    collection_timestamp = models.DateTimeField(**opt)
    noaa_site = models.CharField(max_length=16, **ch_opt, verbose_name='NOAA Site')  # noqa: E501
    env_broad_scale = models.CharField(max_length=64, **ch_opt)
    env_local_scale = models.CharField(max_length=64, **ch_opt)
    env_medium = models.CharField(max_length=64, **ch_opt)
    modified_or_experimental = models.BooleanField(default=False)
    depth = models.CharField(max_length=16, **ch_opt)
    depth_sediment = models.CharField(max_length=16, **ch_opt)
    size_frac_up = models.CharField(max_length=16, **ch_opt)
    size_frac_low = models.CharField(max_length=16, **ch_opt)
    ph = models.CharField(max_length=8, **ch_opt, verbose_name='pH')
    temp = models.CharField(max_length=8, **ch_opt)
    calcium = models.CharField(max_length=8, **ch_opt)
    potassium = models.CharField(max_length=8, **ch_opt)
    magnesium = models.CharField(max_length=8, **ch_opt)
    ammonium = models.CharField(max_length=8, **ch_opt)
    nitrate = models.CharField(max_length=8, **ch_opt)
    phosphorus = models.CharField(max_length=8, **ch_opt)
    diss_oxygen = models.CharField(max_length=8, **ch_opt)
    conduc = models.CharField(max_length=16, **ch_opt)
    secci = models.CharField(max_length=8, **ch_opt)
    turbidity = models.CharField(max_length=8, **ch_opt)
    part_microcyst = models.CharField(max_length=8, **ch_opt)
    diss_microcyst = models.CharField(max_length=8, **ch_opt)
    ext_phyco = models.CharField(max_length=8, **ch_opt)
    chlorophyl = models.CharField(max_length=8, **ch_opt)
    diss_phosp = models.CharField(max_length=8, **ch_opt)
    soluble_react_phosp = models.CharField(max_length=8, **ch_opt)
    ammonia = models.CharField(max_length=8, **ch_opt)
    nitrate_nitrite = models.CharField(max_length=8, **ch_opt)
    urea = models.CharField(max_length=8, **ch_opt)
    part_org_carb = models.CharField(max_length=8, **ch_opt)
    part_org_nitro = models.CharField(max_length=8, **ch_opt)
    diss_org_carb = models.CharField(max_length=8, **ch_opt)
    col_dom = models.CharField(max_length=8, **ch_opt)
    suspend_part_matter = models.CharField(max_length=8, **ch_opt)
    suspend_vol_solid = models.CharField(max_length=8, **ch_opt)
    notes = models.CharField(max_length=512, **ch_opt)

    loader = SampleLoader()

    class Meta:
        default_manager_name = 'objects'

    def __str__(self):
        return self.sample_name or self.sample_id or self.biosample \
            or super().__str__()
