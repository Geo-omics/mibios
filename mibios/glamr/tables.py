from django.urls import reverse
from django.utils.html import escape, format_html, mark_safe

from django_tables2 import Column, Table, TemplateColumn

from mibios.glamr import models as glamr_models
from mibios.omics import models as omics_models


def get_record_url(*args):
    """
    Return URL for an object

    Arguments: <obj> | <<model|model_name> <pk>>
    The object can be passed as the only argument.  Or the model/model name and
    PK must be passed.

    Use this instead of Model.get_absolute_url() because it needs to work on
    models from other apps.
    """
    if len(args) == 1:
        obj = args[0]
        model_name = obj._meta.model_name
        pk = obj.pk
    elif len(args) == 2:
        model, pk = args
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model._meta.model_name
    else:
        raise TypeError(
            'expect either a model instance or model/-name and pk pair'
        )
    return reverse('record', kwargs={'model': model_name, 'pk': pk})


class CompoundAbundanceTable(Table):
    sample = Column(
        linkify=lambda value: get_record_url(value)
    )
    compound = Column(
        linkify=lambda value: get_record_url(value)
    )

    class Meta:
        model = omics_models.CompoundAbundance
        exclude = ['id']


class FunctionAbundanceTable(Table):
    related_genes = Column(
        linkify=lambda record:
            reverse(
                'record_abundance_genes',
                kwargs={
                    'model': record.function._meta.model_name,
                    'pk': record.function.pk,
                    'sample': record.sample.accession,
                },
            ),
        verbose_name='Related genes',
        empty_values=(),  # to trigger render_FOO()
    )

    class Meta:
        model = omics_models.FuncAbundance
        exclude = ['id']

    def render_related_genes(self):
        return 'genes'


class OverViewTable(Table):
    num_samples = TemplateColumn(
        """<a href="{% url 'record_overview_samples' model=table.view_object_model_name pk=table.view_object.pk %}">{{ value }}</a> out of {{ record.total_samples }}""",  # noqa: E501
        verbose_name='Number of samples',
    )
    short = TemplateColumn(
        "{{ record }}",
        linkify=lambda record: get_record_url(record),
        verbose_name='Mini description',
    )

    class Meta:
        model = glamr_models.Dataset
        fields = [
            'num_samples', 'short', 'water_bodies', 'year', 'Institution/PI',
            'sequencing_data_type',
        ]


class OverViewSamplesTable(Table):
    accession = Column(
        linkify=lambda record: get_record_url(record),
        verbose_name='Sample',
    )
    sample_name = Column(verbose_name='Other names')
    dataset = Column(
        linkify=lambda value: get_record_url(value),
        verbose_name='Dataset',
    )

    class Meta:
        model = glamr_models.Sample
        fields = [
            'accession', 'sample_name', 'dataset', 'dataset.water_bodies',
            'date', 'Institution/PI', 'latitude', 'longitude',
        ]


class TaxonAbundanceTable(Table):
    sample = Column(
        linkify=lambda value: get_record_url('sample', value.pk)
    )

    class Meta:
        model = omics_models.TaxonAbundance
        fields = ['sample', 'sum_gene_rpkm']


class DatasetTable(Table):
    scheme = Column(
        empty_values=(),  # so render_foo can still take over for blank scheme
        verbose_name='Description',
        linkify=True,
        attrs={
            'showFieldTitle': False,
            'cardTitle': True,
            'navID': "scheme-sort",
        }
    )
    samples = Column(
        verbose_name='Available samples',
        attrs={
            'showFieldTitle': False,
            'cardTitle': False,
            'defaultSort': True,
            'navID': "samples-sort",
        }
    )
    reference = Column(
        linkify=lambda value: getattr(value, 'doi'),
        attrs={
            'showFieldTitle': True,
            'cardTitle': False,
            'navID': "reference-sort",
        }
    )
    water_bodies = Column(
        verbose_name='Water bodies',
        attrs={
            'showFieldTitle': True,
            'cardTitle': False,
            'navID': "water_bodies-sort",
        }
    )
    material_type = Column(
        attrs={
            'showFieldTitle': True,
            'cardTitle': False,
            'navID': "material_type-sort",
        }
    )
    sample_type = Column(
        empty_values=(),
        verbose_name='Sample type',
        attrs={
            'showFieldTitle': True,
            'cardTitle': False,
            'navID': "sample_type-sort",
        }
    )
    external_urls = Column(
        verbose_name='External links',
        attrs={
            'showFieldTitle': True,
            'cardTitle': False,
            'navID': "external_urls-sort",
        }
    )

    class Meta:
        empty_text = 'No dataset / study information available'
        template_name = 'glamr/table_cards.html'
        attrs = {
            "id": "overview-table",
            "class": "table table-hover",
        }

    def render_scheme(self, value, record):
        r = record
        scheme = r.scheme or r.short_name or r.bioproject \
            or r.jgi_project or r.gold_id
        if not scheme:
            scheme = (f'(no description available) '
                      f'id:{record.dataset_id}/pk:{record.pk}')
        # capitalize just the first letter; leave other characters as they are:
        return scheme[0].upper() + scheme[1:]

    def render_external_urls(self, value, record):
        # value is a list of tuples (accession, url)
        items = []
        for accession, url in value:
            if url:
                items.append(
                    format_html('<a href="{}" class="card-link">{}</a>', url, accession)  # noqa: E501
                )
            else:
                items.append(escape(accession))
        return mark_safe(' '.join(items))

    def render_sample_type(self, record):
        values = set((
            i.sample_type
            for i in record.sample_set.all()
            if i.sample_type
        ))
        values = sorted(values)

        if record.sequencing_target:
            values.append(record.sequencing_target)
        if record.size_fraction:
            values.append(record.size_fraction)
        return ' / '.join(values)

    def render_samples(self, record):
        if record.sample_count <= 0:
            return mark_safe('<div class="btn btn-primary disabled mb-1">No samples</div>')  # noqa: E501

        url = record.get_samples_url()
        return mark_safe(f'<a href="{url}" class="btn btn-primary mb-1">{record.sample_count} available samples</a>')  # noqa: E501


def get_sample_url(sample):
    """ linkify helper for SampleTable """
    return reverse('sample_detail', args=[sample.pk])


class SingleColumnRelatedTable(Table):
    """ Table showing *-to-many related records in single column """
    objects = Column(
        verbose_name='Related records',
        linkify=lambda record: get_record_url(record),
        empty_values=(),  # triggers render_objects()
    )

    def render_objects(self, record):
        return str(record)


class SampleTable(Table):
    sample_name = Column(
        verbose_name='Sample Name/ID',
        # linkify=lambda record: get_sample_url(record),
        empty_values=[],
        linkify=lambda record: reverse('sample', args=[record.pk]),
    )
    geo_loc_name = Column(
        empty_values=[],
        verbose_name='Location / site',
    )
    sample_type = Column()
    dataset = Column(
        linkify=lambda value: reverse('dataset', args=[value.pk]),
        verbose_name='Dataset',
    )

    class Meta:
        model = glamr_models.Sample
        fields = [
            'amplicon_target',
            'collection_timestamp', 'latitude', 'longitude',
        ]
        sequence = ['sample_name', 'sample_type', '...']
        empty_text = 'There are no samples associated with this dataset'
        attrs = {
            "class": "table table-hover",
        }

    def render_sample_name(self, record):
        return str(record)

    def render_geo_loc_name(self, record):
        items = [record.geo_loc_name, record.noaa_site]
        return ' / '.join([i for i in items if i])

    def render_collection_timestamp(self, record):
        return record.format_collection_timestamp()
