from django.urls import reverse
from django.utils.html import escape, format_html, mark_safe

from django_tables2 import Column, Table as Table0, TemplateColumn

from mibios.glamr import models as glamr_models
from mibios.omics import models as omics_models

from .utils import get_record_url


class Table(Table0):
    def __init__(self, *args, view=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = view


def linkify_record(record):
    """
    linkify helper to get record URLs

    Usage:

    class MyTable(Table):
        some_field = Column(..., linkify=linkify_record, ...)

    This should link a column to the record's single object view.
    """
    # The linkify magic senses the signature, so the argument is 'record', that
    # should get us a model instance passed.
    return get_record_url(record)


def linkify_value(value):
    """
    linkify helper to get the value's URLs

    Usage:

    class MyTable(Table):
        fk_field = Column(..., linkify=linkify_value, ...)

    This should link a column based on a foreign key field to the right single
    object view.
    """
    # The linkify magic senses the signature, so the argument is 'value', that
    # should get us the FK target's model instance passed.
    return get_record_url(value)


class CompoundAbundanceTable(Table):
    sample = Column(linkify=linkify_value)
    compound = Column(linkify=linkify_value)

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
        linkify=linkify_record,
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
        linkify=linkify_record,
        verbose_name='Sample',
    )
    sample_name = Column(verbose_name='Other names')
    dataset = Column(
        linkify=linkify_value,
        verbose_name='Dataset',
    )

    class Meta:
        model = glamr_models.Sample
        fields = [
            'accession', 'sample_name', 'dataset', 'dataset.water_bodies',
            'date', 'Institution/PI', 'latitude', 'longitude',
        ]


class TaxonAbundanceTable(Table):
    sample = Column(linkify=linkify_value)

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
        if not hasattr(record, 'sample_count'):
            # sample_count is a queryset annotation, it may be missing
            return ''
        if record.sample_count <= 0:
            return mark_safe('<div class="btn btn-primary disabled mb-1">No samples</div>')  # noqa: E501

        if hasattr(self.view, 'conf'):
            conf = self.view.conf.shift('sample', reverse=True)
            conf.filter['dataset_id'] = record.pk
            url = reverse('filter_result', kwargs=dict(model='sample'))
            url = url + '?' + conf.url_query()
        else:
            url = record.get_samples_url()

        return format_html(
            '<a href="{url}" class="btn btn-primary mb-1">'
            '{count} samples</a>',
            url=url,
            count=record.sample_count,
        )


class SingleColumnRelatedTable(Table):
    """ Table showing *-to-many related records in single column """
    objects = Column(
        verbose_name='Related records',
        linkify=linkify_record,
        empty_values=(),  # triggers render_objects()
        orderable=False,
    )

    def render_objects(self, record):
        return str(record)


class SampleTable(Table):
    sample_name = Column(
        verbose_name='Sample Name/ID',
        linkify=linkify_record,
        empty_values=[],
    )
    geo_loc_name = Column(
        empty_values=[],
        verbose_name='Location / site',
    )
    sample_type = Column()
    dataset = Column(
        linkify=linkify_value,
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
