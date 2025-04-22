from django.db.models import Count
from django.http import Http404
from django.views.generic import TemplateView

from django_tables2 import SingleTableView

from . import tables
from .models import Dataset, Host, Sample


class DetailView(TemplateView):
    template_name = 'hamb/detail.html'

    model = None
    fields = None
    name_field = None

    def get_context_data(self, **ctx):
        try:
            self.obj = self.model.objects.get(pk=self.kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404(f'no such {self.model._meta.modelname}')

        ctx = super().get_context_data(**ctx)
        items = []
        items.append((
            self.model._meta.verbose_name,
            getattr(self.obj, self.name_field),
            None,
        ))
        for i in self.fields:
            field = self.model._meta.get_field(i)
            value = getattr(self.obj, field.name)
            if field.many_to_one:
                url = self.obj.get_absolute_url()
            else:
                url = None
            items.append((field.verbose_name, value, url))

        ctx['items'] = items
        return ctx


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
