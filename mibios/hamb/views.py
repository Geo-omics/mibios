from itertools import chain

from django.db.models import Count
from django.http import Http404
from django.urls import reverse
from django.utils.functional import cached_property
from django.views.generic import DetailView as DetailView0

from django_tables2 import SingleTableView

from mibios.ncbi_taxonomy.models import TaxNode
from mibios.omics.models import ASV, ASVAbundance
from . import filters, tables
from .managers import get_taxon_asv
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


class ASVListing(Listing):
    model = ASV
    table_class = tables.ASVTable

    def get_queryset(self):
        qs = super().get_queryset()
        self.filter = filters.ASVFilter(self.request.GET, queryset=qs)
        return self.filter.qs


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


class DetailView(DetailView0):
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

    def get_items(self):
        """
        Get the details

        Returns a list of 3-tuples: field_name, value, URL.
        """
        items = []
        if self.name_field:
            items.append((
                self.model._meta.verbose_name,
                getattr(self.object, self.name_field),
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
            value = getattr(self.object, field.name)
            if value is None:
                value = ''
            if field.many_to_one:
                url = self.get_other_object_url(value)
            else:
                url = None
            items.append((field.verbose_name, value, url))
        return items

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'] = self.get_items()
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

    def get_items(self):
        items = super().get_items()
        asvnum = self.object.asv_number
        items.append((
            'Abundance',
            self.object.asvabundance_set.count(),
            reverse('asv_abund_list', kwargs={'asvnum': asvnum}),
        ))
        return items


class DatasetDetail(DetailView):
    model = Dataset
    fields = []
    name_field = 'label'

    def get_items(self):
        items = super().get_items()
        items.append((
            'samples',
            self.object.sample_set.count(),
            reverse('sample_list') + f'?dataset={self.object.pk}',
        ))
        return items


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

    def get_items(self):
        items = super().get_items()
        items.append((
            'samples',
            self.object.sample_set.count(),
            reverse('host_sample_list', kwargs={'pk': self.object.pk}),
        ))
        return items


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

    def get_items(self):
        items = super().get_items()
        items.append((
            'ASV Abundance',
            self.object.asv_abundance.count(),
            reverse('sample_abund_list', kwargs={'pk': self.object.pk}),
        ))
        return items


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


class TaxBrowserMixin:
    template_insert_name = 'hamb/tax_browser.html'
    """ add this to implementing template """

    @cached_property
    def node(self):
        """ query DB for node and set instance attribute """
        # inject default for alternative root url
        taxid = self.kwargs.get('taxid', 1)
        try:
            return TaxNode.objects.get(taxid=taxid)
        except TaxNode.DoesNotExist as e:
            raise Http404('no such taxon') from e

    def get_data(self, node):
        """
        Get taxonomy browser data

        These are two lists of 3-tuples: TaxNode, inclusive ASVs, pure ASVs.
        Use the pre-compiled data from managers module for the inclusive ASV
        counts.
        """
        lineage = node.lineage
        children = node.children.all().order_by('name')
        # "pure" ASVs for taxon
        counts = dict(
            TaxNode.objects
            .filter(pk__in=[i.pk for i in chain(lineage, children)])
            .annotate(Count('asv'))
            .values_list('pk', 'asv__count')
        )
        return (
            [(i, len(get_taxon_asv(i)), counts[i.pk]) for i in lineage],
            [(i, len(get_taxon_asv(i)), counts[i.pk]) for i in children],
        )

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['lineage'], ctx['children'] = self.get_data(self.node)
        ctx['tax_browser_insert'] = self.template_insert_name
        return ctx


class TaxonDetail(TaxBrowserMixin, DetailView):
    template_name = 'hamb/taxon_detail.html'
    model = TaxNode
    name_field = 'name'
    fields = ['taxid', 'parent']

    def get_object(self):
        return self.node

    @classmethod
    def get_object_url(cls, obj):
        return reverse('taxon_detail', kwargs={'taxid': obj.taxid})

    def get_items(self):
        items = super().get_items()
        items.append((
            'ASVs',
            self.object.asv_set.count(),
            reverse('asv_list') + f'?taxon__taxid={self.object.taxid}',
        ))
        return items
