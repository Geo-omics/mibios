from django.urls import reverse
from django.utils.html import escape, format_html, mark_safe

from django_tables2 import Column, Table as Table0, TemplateColumn

from mibios.glamr import models as glamr_models
from mibios.ncbi_taxonomy.models import TaxName, TaxNode
from mibios.omics import models as omics_models
from mibios.umrad.models import UniRef100

from .utils import get_record_url


class Table(Table0):
    def __init__(self, data=None, view=None, **kwargs):
        self.view = view
        data = self.customize_queryset(data)
        super().__init__(data=data, **kwargs)

    def customize_queryset(self, qs):
        """
        Table-specific mangling of the queryset.

        Override this method as needed, e.g. add some annotations or run
        select_related() for those additional columns.
        """
        return qs


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


class AboutHistoryTable(Table):
    class Meta:
        # NOTE on order: Most recent version should go on top of page.  But
        # can't order by PK as id column is excluded.  So as unpublished is
        # NULL it'll go first with Postgres (good for production) but last with
        # sqlite (acceptable in development.)
        order_by = ('-when_published',)
        model = glamr_models.AboutInfo
        exclude = ['id', 'generation']
        sequence = ['...', 'comment', 'src_version']


class CompoundAbundanceTable(Table):
    sample = Column(linkify=linkify_value)
    compound = Column(linkify=linkify_value)

    class Meta:
        model = omics_models.CompoundAbundance
        exclude = ['id']


class DBInfoTable(Table):
    num_rows = Column(
        attrs={'td': {'align': 'right'}},
    )
    storage_size = Column(
        empty_values=(),  # triggers render_objects()
        attrs={'td': {'align': 'right'}},
    )

    GB = 1024 * 1024 * 1024

    class Meta:
        # Using model pg_class here, but the model dbstat is similar enough, so
        # this table class should work for postgresql as well as sqlite.  When
        # display with sqlite, django_tables2 will show a warning about the
        # mismatch.
        model = glamr_models.pg_class
        exclude = ('num_pages',)
        sequence = ['...', 'storage_size']
        order_by = 'name'

    def render_num_rows(self, value, record):
        if value < 0:
            # -1 indicates no info
            return ''

        # insert commas to separate thousands, assumes positive integer
        value = str(int(value))
        ncom = (len(value) - 1) // 3  # number of commas
        digits = list(value)
        ii = -3  # index in digit list where to insert right-most comma
        for _ in range(ncom):
            digits.insert(ii, ',')
            ii = ii - 4  # next comma goes 3 digits to the left + prev comma
        return ''.join(digits)

    def render_storage_size(self, value, record):
        if record.num_pages < 0:
            return ''
        psize = self._meta.model.PAGE_SIZE
        return f'{record.num_pages * psize / self.GB:.1f}'

    def order_storage_size(self, qs, is_descending):
        flag = 'num_pages'
        if is_descending:
            flag = '-' + flag
        return (qs.order_by(flag), True)


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


class ReadAbundanceTable(Table):
    sample = Column(linkify=linkify_value)
    ref = Column(linkify=linkify_value, verbose_name='Function/reference')

    class Meta:
        model = omics_models.ReadAbundance
        order_by = ['-target_cov']
        exclude = ['id']

    def customize_queryset(self, qs):
        try:
            is_export = self.view.export_check() is not None
        except Exception:
            is_export = False
        if not is_export:
            # For normal HTML table get the function names, don't need or want
            # those for export
            qs = qs.prefetch_related('ref__function_names')
        return qs

    def render_ref(self, value):
        funcs = list(value.function_names.all())
        if funcs:
            value = str(funcs[0])
            for i in funcs[1:]:
                if len(value) > 80:
                    value += ', ...'
                    break
                value += f', {i}'
        else:
            # just the uniref100 accession
            value = str(value)
            if value.startswith('UNIREF100_'):
                value = 'Uniref100: {value.removeprefix("UNIREF100_")}'
        return value

    def as_values(self):
        """
        Table export, re-implemented for speed with large data
        """
        qs = self.data.data.values_list(*self.sequence)

        # Get lookup tables for samples and/or reference,
        # this depends on whether we have those fields
        try:
            sample_key = self.sequence.index('sample')
        except ValueError:
            sample_key = None
        try:
            ref_key = self.sequence.index('ref')
        except ValueError:
            ref_key = None

        sample_pks = set()
        ref_pks = set()
        for row in qs:
            if sample_key is not None:
                sample_pks.add(row[sample_key])
            if ref_key is not None:
                ref_pks.add(row[ref_key])
        # TODO: these maps could/should be globally cached?
        replacer = [None] * len(self.sequence)
        if sample_key is not None:
            # add sample PK to sample name map
            replacer[sample_key] = {
                obj.pk: str(obj)
                for obj
                in glamr_models.Sample.objects.filter(pk__in=sample_pks)
            }
        if ref_key is not None:
            # add ref PK to ref text value mapping:
            replacer[ref_key] = {
                obj.pk: str(obj)
                for obj in UniRef100.objects.filter(pk__in=ref_pks)
            }

        def convert(row):
            for mapping, value in zip(replacer, row):
                if mapping is None:
                    yield value
                else:
                    yield mapping[value]

        for row in qs:
            yield tuple(convert(row))


class TaxNodeTable(Table):
    taxid = Column(linkify=linkify_record)
    parent = Column(linkify=linkify_value)

    class Meta:
        model = TaxNode
        fields = ('taxid', 'rank', 'name', 'parent')


class TaxonAbundanceTable(Table):
    sample = Column(linkify=linkify_value)
    taxon = Column(linkify=linkify_value, verbose_name='Tax ID')
    rank = Column(accessor='taxon__rank')
    tax_name = Column(accessor='taxon', orderable=False,
                      verbose_name='Tax Name')

    class Meta:
        model = omics_models.TaxonAbundance
        fields = ['sample', 'taxon', 'rank', 'tax_name', 'tpm']
        order_by = ['-tpm']

    def customize_queryset(self, qs):
        qs = qs.select_related('taxon')
        qs = qs.only('sample_id', 'taxon__taxid', 'taxon__rank', 'tpm')
        return qs

    def render_taxon(self, value):
        return value.taxid

    def render_tax_name(self, value):
        # value is TaxNode obj
        return value.name

    def as_values(self):
        """
        Table export, re-implemented for speed with large data
        """
        # collect tax names and sample names for fast lookup
        # TODO: these maps could be cached globally, saving couple seconds here
        names = TaxName.objects.filter(name_class=TaxName.NAME_CLASS_SCI)
        names = names.only('node_id', 'name')
        tax_name_map = dict(names.values_list('node_id', 'name'))
        del names
        sample_name_map = {
            i.pk: str(i)
            for i in glamr_models.Sample.objects.all()
        }

        # we assume data is a QuerySet
        qs = self.data.data

        # Output fields, order etc. need to be manually kept in sync with table
        # declarations.
        fields = ['sample_id', 'taxon_id', 'taxon__taxid', 'taxon__rank',
                  'tpm']
        for spk, tpk, taxid, rank, tpm in qs.values_list(*fields):
            if tpk is None:
                taxid = ''
                rank = ''
                name = 'unclassified'  # TODO: also show unclass on HTML output
            else:
                name = tax_name_map[tpk]

            yield [
                sample_name_map[spk],
                taxid,
                rank,
                name,
                tpm,
            ]

        return


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
    dataset = Column(
        linkify=linkify_value,
        verbose_name='Dataset',
    )

    class Meta:
        model = glamr_models.Sample
        sequence = ['sample_name', 'sample_type', '...', 'dataset']
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
