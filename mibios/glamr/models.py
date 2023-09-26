"""
GLAMR-specific modeling
"""
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.indexes import GinIndex, GistIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.urls import reverse

from mibios.omics.models import AbstractDataset, AbstractSample
from mibios.umrad.fields import AccessionField
from mibios.umrad.models import Model
from mibios.umrad.model_utils import ch_opt, fk_opt, fk_req, uniq_opt, opt

from .fields import OptionalURLField
from .load import (
    DatasetLoader, DatasetManager, ReferenceLoader, SampleLoader,
    SampleManager, SearchableManager, UniqueWordManager,
)
from .queryset import (
    DatasetQuerySet, SampleQuerySet, SearchableQuerySet, UniqueWordQuerySet,
)


class Dataset(AbstractDataset):
    """
    A collection of related samples, e.g. a study or project
    """
    dataset_id = AccessionField(
        # overrides abstract parent field
        unique=True,
        verbose_name='Dataset ID',
        help_text='GLAMR accession to data set/study/project',
    )
    private = models.BooleanField(
        default=False,
        help_text='hide this record and related samples from public view',
    )
    reference = models.ForeignKey('Reference', **fk_opt)
    # project IDs: usually a single accession, but can be ,-sep lists or even
    # other text
    bioproject = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='NCBI BioProject',
    )
    jgi_project = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='JGI Project ID',
    )
    gold_id = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='GOLD ID',
    )
    scheme = models.TextField(
        **ch_opt,
        verbose_name='location and sampling scheme',
    )
    material_type = models.TextField(
        **ch_opt,
    )
    water_bodies = models.TextField(
        **ch_opt,
        help_text='list or description of sampled bodies of water',
    )
    primers = models.TextField(
        max_length=32,
        **ch_opt,
    )
    sequencing_target = models.TextField(
        max_length=32,
        **ch_opt,
    )
    sequencing_platform = models.TextField(
        max_length=32,
        **ch_opt,
    )
    size_fraction = models.TextField(
        max_length=32,
        **ch_opt,
        verbose_name='size fraction(s)',
        help_text='e.g.: >0.22µm or 0.22-1.6µm',
    )
    note = models.TextField(**ch_opt)

    accession_fields = ('dataset_id', )

    objects = DatasetManager.from_queryset(DatasetQuerySet)()
    loader = DatasetLoader()

    class Meta:
        default_manager_name = 'objects'

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

        s = ' - '.join(filter(None, [scheme, ref])) or self.short_name \
            or super().__str__()

        if settings.INTERNAL_DEPLOYMENT:
            s = f'{s} ({self.dataset_id})'

        return s

    URL_TEMPLATES = {
        'bioproject': 'https://www.ncbi.nlm.nih.gov/bioproject/{}',
        'gold_id': {
            '^Ga': 'https://gold.jgi.doe.gov/analysis_project?id={}',
            '^Gp': 'https://gold.jgi.doe.gov/project?id={}',
            '^Gs': 'https://gold.jgi.doe.gov/study?id={}',
            '^Gb': 'https://gold.jgi.doe.gov/biosample?id={}',
        },
    }

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
        return reverse('dataset', args=[self.pk])

    def get_samples_url(self):
        return reverse('dataset_sample_list', args=[self.pk])


class Reference(Model):
    """
    A journal article or similar, primary reference for a data set
    """
    reference_id = AccessionField(prefix='paper_')
    short_reference = models.TextField(
        # this field is required
        max_length=32,
        help_text='short reference',
    )
    authors = models.TextField(
        **ch_opt,
        help_text='author listing',
    )
    title = models.TextField(**ch_opt)
    abstract = models.TextField(**ch_opt)
    key_words = models.TextField(**ch_opt)
    publication = models.TextField(max_length=64, **ch_opt)
    doi = OptionalURLField(**uniq_opt)

    loader = ReferenceLoader()

    class Meta:
        pass

    def __str__(self):
        maxlen = 60
        value = f'{self.short_reference} "{self.title}"'
        if len(value) > maxlen:
            value = value[:maxlen]
            # rm last word, add ellipsis
            value = ' '.join(value.split(' ')[:-1]) + '[\u2026]"'
        if settings.INTERNAL_DEPLOYMENT:
            value = f'{value} ({self.reference_id})'
        return value

    def get_absolute_url(self):
        return reverse('reference', args=[self.pk])


class Sample(AbstractSample):
    DATE_ONLY = 'date_only'
    YEAR_ONLY = 'year_only'
    MONTH_ONLY = 'month_only'
    FULL_TIMESTAMP = ''
    PARTIAL_TS_CHOICES = (
        (DATE_ONLY, DATE_ONLY),
        (YEAR_ONLY, YEAR_ONLY),
        (MONTH_ONLY, MONTH_ONLY),
        (FULL_TIMESTAMP, FULL_TIMESTAMP),
    )

    project_id = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='NCBI BioProject',
        help_text='Project accession, e.g. NCBI bioproject',
    )
    biosample = models.TextField(max_length=32, **ch_opt)
    gold_analysis_id = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='GOLD analysis ID',
    )
    gold_seq_id = models.TextField(
        max_length=32,
        **ch_opt,
        verbose_name='GOLD sequencing project ID',
    )
    jgi_study = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='JGI study ID',
    )
    jgi_biosample = models.TextField(
        max_length=32, **ch_opt,
        verbose_name='JGI biosample ID',
    )
    geo_loc_name = models.TextField(max_length=64, **ch_opt)
    gaz_id = models.TextField(max_length=32, **ch_opt, verbose_name='GAZ id')
    latitude = models.TextField(max_length=16, **ch_opt)
    longitude = models.TextField(max_length=16, **ch_opt)
    # timestamp: expect ISO8601 formats plus yyyy and yyyy-mm
    collection_timestamp = models.DateTimeField(**opt)
    # Indicate missing time or partial non-ISO6801 dates: e.g. 2013 or 2013-08
    collection_ts_partial = models.CharField(
        max_length=10,
        choices=PARTIAL_TS_CHOICES,
        default=FULL_TIMESTAMP,
        blank=True,
    )
    noaa_site = models.TextField(max_length=16, **ch_opt, verbose_name='NOAA Site')  # noqa: E501
    env_broad_scale = models.TextField(max_length=32, **ch_opt)
    env_local_scale = models.TextField(max_length=32, **ch_opt)
    env_medium = models.TextField(max_length=32, **ch_opt)
    keywords = models.TextField(max_length=32, **ch_opt)
    depth = models.TextField(max_length=16, **ch_opt)
    depth_sediment = models.TextField(max_length=16, **ch_opt)
    depth_location = models.DecimalField(max_digits=7, decimal_places=2, **opt)
    size_frac_up = models.TextField(max_length=16, **ch_opt)
    size_frac_low = models.TextField(max_length=16, **ch_opt)
    ph = models.TextField(max_length=8, **ch_opt, verbose_name='pH')
    temp = models.TextField(max_length=8, **ch_opt)
    calcium = models.TextField(max_length=8, **ch_opt)
    potassium = models.TextField(max_length=8, **ch_opt)
    phosphate = models.TextField(max_length=8, **ch_opt)
    magnesium = models.TextField(max_length=8, **ch_opt)
    ammonium = models.TextField(max_length=8, **ch_opt)
    nitrate = models.TextField(max_length=8, **ch_opt)
    diss_oxygen = models.TextField(max_length=8, **ch_opt)
    conduc = models.TextField(max_length=16, **ch_opt)
    secchi = models.TextField(max_length=8, **ch_opt)
    turbidity = models.TextField(max_length=8, **ch_opt)
    part_microcyst = models.TextField(max_length=8, **ch_opt)
    diss_microcyst = models.TextField(max_length=8, **ch_opt)
    ext_phyco = models.TextField(max_length=8, **ch_opt)
    ext_microcyst = models.TextField(max_length=8, **ch_opt)
    ext_anatox = models.TextField(max_length=8, **ch_opt)
    chlorophyl = models.TextField(max_length=8, **ch_opt)
    total_phos = models.TextField(max_length=8, **ch_opt)
    diss_phos = models.TextField(max_length=8, **ch_opt)
    soluble_react_phos = models.TextField(max_length=8, **ch_opt)
    ammonia = models.TextField(max_length=8, **ch_opt)
    nitrate_nitrite = models.TextField(max_length=8, **ch_opt)
    urea = models.TextField(max_length=8, **ch_opt)
    part_org_carb = models.TextField(max_length=8, **ch_opt)
    part_org_nitro = models.TextField(max_length=8, **ch_opt)
    diss_org_carb = models.TextField(max_length=8, **ch_opt)
    col_dom = models.TextField(max_length=8, **ch_opt)
    h2o2 = models.TextField(max_length=8, **ch_opt)
    suspend_part_matter = models.TextField(max_length=8, **ch_opt)
    suspend_vol_solid = models.TextField(max_length=8, **ch_opt)
    microcystis_count = models.PositiveIntegerField(**opt)
    planktothrix_count = models.PositiveIntegerField(**opt)
    anabaena_d_count = models.PositiveIntegerField(**opt)
    cylindrospermopsis_count = models.PositiveIntegerField(**opt)
    ice_cover = models.PositiveSmallIntegerField(**opt)
    chlorophyl_fluoresence = models.DecimalField(max_digits=5, decimal_places=2, **opt)  # noqa:E501
    sampling_device = models.TextField(max_length=32, **ch_opt)
    modified_or_experimental = models.BooleanField(default=False)
    is_isolate = models.BooleanField(**opt)
    is_neg_control = models.BooleanField(**opt)
    is_pos_control = models.BooleanField(**opt)
    samp_vol_we_dna_ext = models.DecimalField(max_digits=10, decimal_places=3, **opt)  # noqa: E501
    filt_duration = models.DurationField(**opt)
    qPCR_total = models.PositiveIntegerField(**opt)
    qPCR_mcyE = models.PositiveIntegerField(**opt)
    qPCR_sxtA = models.PositiveIntegerField(**opt)
    silicate = models.TextField(max_length=32, **ch_opt)
    tot_nit = models.TextField(max_length=32, **ch_opt)
    green_algae = models.TextField(max_length=32, **ch_opt)
    bluegreen = models.TextField(max_length=32, **ch_opt)
    diatoms = models.TextField(max_length=32, **ch_opt)
    crypto = models.TextField(max_length=32, **ch_opt)
    tot_microcyst_lcmsms = models.TextField(max_length=32, **ch_opt)
    attenuation = models.TextField(max_length=32, **ch_opt)
    transmission = models.TextField(max_length=32, **ch_opt)
    par = models.DecimalField(max_digits=8, decimal_places=2, **opt)
    sky = models.TextField(max_length=32, **ch_opt)
    wave_height = models.TextField(max_length=32, **ch_opt)
    wind_speed = models.TextField(max_length=32, **ch_opt)
    sortchem = models.TextField(max_length=32, **ch_opt)
    phyco_fluoresence = models.TextField(max_length=32, **ch_opt)

    notes = models.TextField(**ch_opt)

    objects = SampleManager.from_queryset(SampleQuerySet)()
    loader = SampleLoader()

    class Meta:
        default_manager_name = 'objects'

    def __str__(self):
        value = self.sample_name or self.biosample or ''
        if value:
            if self.sample_id and settings.INTERNAL_DEPLOYMENT:
                value = f'{value} ({self.sample_id})'
        return value or self.sample_id or super().__str__()

    def format_collection_timestamp(self):
        """
        format partial timestamp

        Returns a str.
        """
        ts = self.collection_timestamp
        match self.collection_ts_partial:
            case self.DATE_ONLY:
                return ts.date().isoformat()
            case self.YEAR_ONLY:
                return ts.strftime('%Y')
            case self.MONTH_ONLY:
                return ts.strftime('%B %Y')
            case _:
                if ts is None:
                    return ''
                else:
                    # FULL_TIMESTAMP
                    return ts.astimezone().isoformat()

    URL_TEMPLATES = {
        'project_id': 'https://www.ncbi.nlm.nih.gov/bioproject/{}',
        'biosample': 'https://www.ncbi.nlm.nih.gov/biosample/{}',
        'gold_analysis_id': Dataset.URL_TEMPLATES['gold_id']['^Ga'],
        'gold_seq_id': Dataset.URL_TEMPLATES['gold_id']['^Gp'],
        'jgi_study_id': Dataset.URL_TEMPLATES['gold_id']['^Gs'],
        'jgi_biosample_id': Dataset.URL_TEMPLATES['gold_id']['^Gb'],
        'sra_accession': {
            '^SRR': 'https://trace.ncbi.nlm.nih.gov/Traces/?view=run_browser&acc={}&display=metadata',  # noqa:E501
            None: 'https://www.ncbi.nlm.nih.gov/sra/?term={}',
        }
    }


class Searchable(models.Model):
    """
    Text that is subject to full-text search

    To populate the table run Searchable.objects.reindex(). Under postgresql
    the searchvector field should be "GENERATED ALWAYS AS" which means we can't
    include the field in INSERTs from django.  A workable hack is to remove
    searchvector from Meta.concrete_fields at runtime.

    Under non-postgresql backends full-text search works, however slowly, via
    icontains lookup or similar on the text column.
    """
    text = models.TextField(max_length=32)
    has_hit = models.BooleanField(default=False)
    content_type = models.ForeignKey(ContentType, **fk_req)
    field = models.CharField(max_length=100)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')
    searchvector = SearchVectorField(null=True)

    objects = SearchableManager.from_queryset(SearchableQuerySet)()

    class Meta:
        indexes = [
            GinIndex(
                fields=['searchvector'],
                # bake in the name, else makemigrations will re-create the
                # index with a different name, with changed hash part (it's a
                # mystery)
                name='glamr_searc_searchv_f71dcf_gin',
            )
        ]

    def __str__(self):
        if len(self.text) > 50:
            return f'{self.field}:{self.text[:45]}[...]'
        else:
            return f'{self.field}:{self.text}'


class UniqueWord(models.Model):
    """
    A distinct word in the Searchable text field

    We use this for spelling suggestions with postgresql's trigram similarity.
    Populate the table with UniqueWord.objects.reindex() after Searchable is
    updated.  Then find spelling suggestions with
    UniqueWord.objects.all().suggest(word).
    """
    word = models.TextField()

    objects = UniqueWordManager.from_queryset(UniqueWordQuerySet)()

    class Meta:
        indexes = [
            GistIndex(
                name='term_trigram_idx',
                fields=['word'],
                opclasses=['gist_trgm_ops'],
            ),
        ]

    def __str__(self):
        return self.word
