from django.db.models import Count
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView

from django_tables2 import SingleTableView

from mibios.ncbi_taxonomy.models import TaxNode
from mibios.omics.models import ASV, ASVAbundance
from . import tables
from .models import Dataset, Host, Sample


class ASVAbundanceListing(SingleTableView):
    template_name = 'hamb/list.html'
    model = ASVAbundance
    table_class = tables.ASVAbundanceTable


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


class DatasetDetail(DetailView):
    model = Dataset
    fields = []
    name_field = 'label'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['items'].append(('samples', self.obj.sample_set.count(), None))
        return ctx


class DatasetListing(SingleTableView):
    template_name = 'hamb/list.html'
    model = Dataset
    table_class = tables.DatasetTable

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.annotate(sample_count=Count('sample'))
        return qs


class HostDetail(DetailView):
    model = Host
    name_field = 'label'
    fields = ['common_name', 'age_years', 'description', 'health_state']


class SampleDetail(DetailView):
    model = Sample
    name_field = 'label'
    fields = [
        'sample_type', 'sra_accession', 'amplicon_target', 'biosample',
        'source_material', 'control',
    ]


class SampleListing(SingleTableView):
    template_name = 'hamb/list.html'
    model = Sample
    table_class = tables.SampleTable

    def XXX_get_queryset(self):
        ...


class TaxBrowser(TemplateView):
    ...


class TaxonDetail(DetailView):
    model = TaxNode
    name_field = 'name'
    fields = ['taxid', 'parent']

    def get_object(self):
        return TaxNode.objects.get(taxid=self.kwargs['taxid'])

    @classmethod
    def get_object_url(cls, obj):
        return reverse('taxon_detail', kwargs={'taxid': obj.taxid})
