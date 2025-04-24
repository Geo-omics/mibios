from django.db.models import Count
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView

from django_tables2 import SingleTableView

from mibios.ncbi_taxonomy.models import TaxNode
from mibios.omics.models import ASV, ASVAbundance
from . import filters, tables
from .models import Dataset, Host, Sample


class Listing(SingleTableView):
    template_name = 'hamb/list.html'
    model = None
    table_class = None
    filter = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['filter'] = self.filter
        ctx['header'] = self.model._meta.verbose_name_plural
        return ctx


class ASVAbundanceListing(Listing):
    model = ASVAbundance
    table_class = tables.ASVAbundanceTable

    def get_queryset(self):
        qs = super().get_queryset()
        self.filter = filters.ASVAbundanceFilter(
            self.request.GET,
            queryset=qs,
        )
        return self.filter.qs


class DetailView(TemplateView):
    template_name = 'hamb/detail.html'

    model = None
    fields = None
    name_field = None

    _detail_view_registry = {}
    """ keep track which view handles which model """

    @classmethod
    def __init_subclass__(cls):
        """ Sub classes run this to register themselves """
        if cls.model in cls._detail_view_registry:
            raise RuntimeError('only on detail view per model is supported')
        cls._detail_view_registry[cls.model] = cls

    @classmethod
    def get_other_object_url(cls, obj):
        """ Get URL for object with other detail view """
        for model, view_class in cls._detail_view_registry.items():
            if isinstance(obj, model):
                return view_class.get_object_url(obj)
        return None

    @classmethod
    def get_object_url(cls, obj):
        """
        Get URL for objects this view is handling

        Override this if the default implementation is not appropriate.
        """
        return obj.get_absolute_url()

    def get_object(self):
        """
        Get the object

        The default implementation assumed that the PK is in the kwargs passed
        from the URL resolver.  Override as needed.

        May raise DoesNotExist for URLs with invalid arguments.
        """
        return self.model.objects.get(pk=self.kwargs['pk'])

    def get_context_data(self, **ctx):
        try:
            self.obj = self.get_object()
        except self.model.DoesNotExist as e:
            raise Http404(f'no such {self.model._meta.modelname}') from e

        ctx = super().get_context_data(**ctx)
        items = []
        if self.name_field:
            items.append((
                self.model._meta.verbose_name,
                getattr(self.obj, self.name_field),
                None,
            ))

        if self.fields:
            fields = [self.model._meta.get_field(i) for i in self.fields]
        else:
            fields = [
                i for i in self.model._meta.get_fields()
                if not i.is_relation  # excl. most relations
                or i.many_to_one  # incl. FKs
            ]

        for field in fields:
            value = getattr(self.obj, field.name)
            if value is None:
                value = ''
            if field.many_to_one:
                url = self.get_other_object_url(value)
            else:
                url = None
            items.append((field.verbose_name, value, url))

        ctx['items'] = items
        return ctx


class ASVDetail(DetailView):
    model = ASV
    fields = ['sequence', 'taxon']
    name_field = 'accession'

    def get_object(self):
        accn = f'{self.model.PREFIX}{self.kwargs["asvnum"]}'
        return self.model.objects.get(accession=accn)

    @classmethod
    def get_object_url(cls, obj):
        return reverse('asv_detail', kwargs={'asvnum': obj.asv_number})

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'].append((
            'Abundance',
            self.obj.asvabundance_set.count(),
            reverse('asv_abund_list', kwargs={'asvnum': self.obj.asv_number}),
        ))
        return ctx


class DatasetDetail(DetailView):
    model = Dataset
    fields = []
    name_field = 'label'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'].append((
            'samples',
            self.obj.sample_set.count(),
            reverse('sample_list') + f'?dataset={self.obj.pk}',
        ))
        return ctx


class DatasetListing(Listing):
    model = Dataset
    table_class = tables.DatasetTable

    def get_queryset(self):
        qs = super().get_queryset()
        self.filter = filters.DatasetFilter(self.request.GET, queryset=qs)
        return self.filter.qs.annotate(sample_count=Count('sample'))


class HostDetail(DetailView):
    model = Host
    name_field = 'label'
    fields = ['common_name', 'age_years', 'description', 'health_state']

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'].append((
            'hosts',
            self.obj.sample_set.count(),
            reverse('host_sample_list', kwargs={'pk': self.obj.pk}),
        ))
        return ctx


class SampleAbundList(ASVAbundanceListing):
    """ ASV abundance for single sample """
    def get_table_kwargs(self):
        return {'exclude': 'sample'}

    def get_queryset(self):
        sample_qs = Sample.objects.select_related('dataset')
        try:
            self.sample = sample_qs.get(pk=self.kwargs['pk'])
        except Sample.DoesNotExist as e:
            raise Http404('no such sample') from e

        qs = super().get_queryset()
        return qs.filter(sample=self.sample)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['header'] = f'{self.sample.dataset}: {self.sample}'
        return ctx


class SampleDetail(DetailView):
    model = Sample
    name_field = 'label'
    fields = [
        'dataset', 'host',
        'sample_type', 'sra_accession', 'amplicon_target', 'biosample',
        'source_material', 'control',
    ]

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'].append((
            'ASV Abundance',
            self.obj.asv_abundance.count(),
            reverse('sample_abund_list', kwargs={'pk': self.obj.pk}),
        ))
        return ctx


class SampleListing(Listing):
    model = Sample
    table_class = tables.SampleTable

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.select_related('dataset', 'host')
        self.filter = filters.SampleFilter(self.request.GET, queryset=qs)
        return self.filter.qs


class HostSampleListing(SampleListing):
    def get_queryset(self):
        try:
            host = Host.objects.get(pk=self.kwargs['pk'])
        except Host.DoesNotExist as e:
            raise Http404('no such host') from e

        qs = super().get_queryset()
        return qs.filter(host=host)


class SingleASVAbundList(ASVAbundanceListing):
    """ ASV abundance for single ASV """
    def get_table_kwargs(self):
        return {'exclude': 'asv'}

    def get_queryset(self):
        accn = f'{ASV.PREFIX}{self.kwargs["asvnum"]}'
        try:
            self.asv = ASV.objects.get(accession=accn)
        except ASV.DoesNotExist as e:
            raise Http404('no such ASV') from e

        qs = super().get_queryset()
        return qs.filter(asv=self.asv)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['header'] = f'{self.asv}'
        return ctx


class TaxBrowser(TemplateView):
    template_name = 'hamb/tax_browser.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # inject default for alternative root url
        taxid = self.kwargs.get('taxid', 1)
        try:
            node = TaxNode.objects.get(taxid=taxid)
        except TaxNode.DoesNotExist as e:
            raise Http404('no such taxon') from e
        ctx['lineage'] = [
            (i.taxid, i.rank, i.name)
            for i in node.lineage
        ]
        ctx['children'] = [
            (i.taxid, i.rank, i.name)
            for i in node.children.all().order_by('name')
        ]
        return ctx


class TaxonDetail(DetailView):
    model = TaxNode
    name_field = 'name'
    fields = ['taxid', 'parent']

    def get_object(self):
        return TaxNode.objects.get(taxid=self.kwargs['taxid'])

    @classmethod
    def get_object_url(cls, obj):
        return reverse('taxon_detail', kwargs={'taxid': obj.taxid})
