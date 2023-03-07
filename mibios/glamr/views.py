from itertools import chain, groupby
from logging import getLogger

from django_tables2 import Column, SingleTableView, TemplateColumn

import pandas

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import FieldDoesNotExist
from django.db import OperationalError
from django.db.models import Count, Field, URLField
from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.generic import DetailView
from django.views.generic.base import TemplateView

from mibios import get_registry
from mibios.data import TableConfig
from mibios.glamr.filters import DatasetFilter
from mibios.glamr.forms import DatasetFilterFormHelper
from mibios.glamr.models import Sample, Dataset
from mibios.models import Q
from mibios.views import ExportBaseMixin, TextRendererZipped
from mibios.omics import get_sample_model
from mibios.omics.models import (
    CompoundAbundance, FuncAbundance, TaxonAbundance
)
from mibios.umrad.models import FuncRefDBEntry
from mibios.omics.models import Gene
from . import models, tables
from .forms import (
    AdvancedSearchForm, QBuilderForm, QLeafEditForm,
)
from .search_utils import get_suggestions

import json


log = getLogger(__name__)


class ExportMixin(ExportBaseMixin):
    query_param = 'export'

    def get_filename(self):
        value = ''
        if hasattr(self, 'object_model'):
            value += self.object_model._meta.model_name
        if hasattr(self, 'object'):
            if value:
                value += '-'
            value += str(self.object)
        if hasattr(self, 'model'):
            if value:
                value += '-'
            value += self.model._meta.model_name
        if value:
            return value
        else:
            return self.__class__.__name__.lower() + '-export'

    def get(self, request, *args, **kwargs):
        if self.export_check():
            return self.export_response()
        else:
            return super().get(request, *args, **kwargs)

    def export_check(self):
        """ Returns wether a file export response is needed """
        return self.query_param in self.request.GET

    def export_response(self):
        """ generate file download response """
        name, suffix, renderer_class = self.get_format()

        response = HttpResponse(content_type=renderer_class.content_type)
        filename = self.get_filename() + suffix
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        renderer_class(response, filename=filename).render(self.get_values())
        return response

    def get_values(self):
        if hasattr(self, 'get_table'):
            return self.get_table().as_values()
        else:
            raise RuntimeError('not implemented')


class BaseFilterMixin:
    """
    Basic filter infrastructure, sufficient to view the filter
    """
    def setup(self, request, *args, model=None, **kwargs):
        super().setup(request, *args, **kwargs)
        self.filter_item_form = None

        if model:
            try:
                self.model = get_registry().models[model]
            except KeyError as e:
                raise Http404(f'no such model: {e}') from e
        self.model_name = self.model._meta.model_name
        if 'search_filter' in request.session \
                and self.model_name == request.session.get('search_model'):
            self.q = Q.deserialize(request.session['search_filter'])
        else:
            # (a) no filter in session yet or
            # (b) data category changed -> so ignore session filter
            self.q = Q()

    def get_queryset(self):
        return super().get_queryset().filter(self.q).distinct()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name'] = self.model._meta.model_name
        ctx['model_verbose_name'] = self.model._meta.verbose_name
        ctx['model_verbose_name_plural'] = self.model._meta.verbose_name_plural
        ctx['editable'] = False
        ctx['qnode'] = self.q
        ctx['qnode_path'] = None
        ctx['col_width'] = 9
        return ctx


class EditFilterMixin(BaseFilterMixin):
    """
    Provide complex filter editor
    """
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.old_q = self.q  # keep backup for revert on error

    def post(self, request, *args, **kwargs):
        self._post(request)
        return super().get(request, *args, **kwargs)

    def _post(self, request):
        """
        do processing of the POST

        In the method body of _post() we mostly do error handling.  There are
        three parts:

            1. validate POST data
            2. process the POST data, apply changes to self.q
            3. verify self.q by passing it to QuerySet.filter() -- without
               hitting the DB

        We return as soon as we catch and handle an error.  Error handling that
        does not result in raising Http404 should revert any state changes
        (e.g. self.q) and give some user feedback. If no errors occur, then the
        session will be updated at the end.
        """
        form = QBuilderForm(data=request.POST)
        if not form.is_valid():
            raise Http404('post request form invalid', form.errors)

        path = form.cleaned_data['path']
        try:
            self.q.resolve_path(path)
        except LookupError:
            log.debug(f'bad path with {request.POST=} {path=}')
            # invalid path, e.g. remove and then resend POST
            # so ignoring this
            messages.add_message(
                request,
                messages.WARNING,
                'Sorry, could not process the request - maybe you went "back" '
                'with your web browser and/or re-submitted a previous request.'
            )
            return

        try:
            action = self.process_post(path)
        except Http404:
            raise
        except Exception as e:
            self.q = self.old_q  # revert changes
            log.error(f'ERROR: {type(e)}: {e} with {request.POST=} {path=}')
            messages.add_message(
                request,
                messages.WARNING,
                # TODO: users should not see errors here
                'Oops: error processing the request'
            )
            return

        try:
            self.model.objects.filter(self.q)
        except Exception as e:
            log.error(f'EDIT Q FILTER: q failed: {type(e)=} {e=} {self.q=}')
            # TODO: eventually the UI should only allow very few things go
            # wrong here, e.g. rhs type/value errors in free text fields.  And
            # those needs to be fixed by the user, so the error messages must
            # be really helpful.
            self.q = self.old_q  # revert changes
            if action == 'apply_leaf_change':
                self.filter_item_form = QLeafEditForm(
                    model=self.model,
                    data=self.request.POST,
                )
                self.filter_item_form.is_valid()
                self.filter_item_form.add_error(None, e)
            else:
                messages.add_message(
                    request,
                    messages.ERROR,
                    f'Sorry, bad filter syndrome: {e.__class__.__name__}: {e}',
                )
            return

        request.session['search_filter'] = self.q.serialize()
        request.session['search_model'] = self.model_name

    def process_post(self, path):
        action = self.request.POST.get('action', None)
        if action == 'rm':
            self.q = self.q.remove_node(path)
        elif action == 'neg':
            self.q = self.q.negate_node(path)
        elif action == 'add':
            self.filter_item_form = QLeafEditForm(
                model=self.model,
                add_mode=True,
                path=path,
            )
        elif action == 'edit':
            lhs, rhs = self.q.resolve_path(path)[-1]
            lhs = lhs.split('__')
            if lhs[-1] in Field.get_lookups():
                lookup = lhs.pop(-1)
            else:
                lookup = 'exact'
            self.filter_item_form = QLeafEditForm(
                model=self.model,
                add_mode=False,
                path=path,
                key='__'.join(lhs),
                lookup=lookup,
                value=rhs,
            )
        elif action == 'apply_leaf_change':
            self.q = self.apply_leaf_changes()
        elif action == 'flip':
            self.q = self.q.flip_node(path)
        else:
            raise Http404('invalid action')
        return action

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['editable'] = True
        return ctx

    def apply_leaf_changes(self):
        form = QLeafEditForm(
            model=self.model,
            data=self.request.POST,
        )
        if not form.is_valid():
            raise Http404('apply changes request form invalid', form.errors)
        lhs = form.cleaned_data['key']
        lhs += '__' + form.cleaned_data['lookup']
        rhs = form.cleaned_data['value']
        path = form.cleaned_data['path']
        if form.cleaned_data['add_mode']:
            return self.q.add_condition(lhs, rhs, path)
        else:
            return self.q.replace_node((lhs, rhs), path)


class ModelTableMixin(ExportMixin):
    """
    Mixin for SingleTableView

    Improves columns for relation fields.  The inheriting view must set
    self.model
    """
    model = None  # model needs to be set by inheriting class
    table_class = None  # triggers the model-based table class creation
    exclude = ['id']  # do not display these fields

    def get_table_kwargs(self):
        kw = {}
        kw['exclude'] = self.exclude
        kw['extra_columns'] = self.get_improved_columns()
        return kw

    def get_improved_columns(self):
        """ make replacements to linkify FK + accession columns """
        cols = []
        try:
            acc_field = self.model.get_accession_field_single()
        except RuntimeError:
            acc_field = None
            col = TemplateColumn(
                """[<a href="{% url 'record' model=model_name pk=record.pk %}">{{ record }}</a>]""",  # noqa:E501
                extra_context=dict(model_name=self.model._meta.model_name),
            )
            cols.append(('record links', col))

        for i in self.model._meta.get_fields():
            if acc_field and i is acc_field:
                col = Column(
                    linkify=lambda record: tables.get_record_url(record)
                )
            elif not i.many_to_one:
                continue
            elif i.name in self.exclude:
                continue
            else:
                # regular FK field
                col = Column(
                    linkify=lambda value: tables.get_record_url(value)
                )
            cols.append((i.name, col))
        return cols


class MapMixin():
    """
    Mixin for views that display samples on a map
    """
    def get_sample_queryset(self):
        """
        Return a queryset of the samples to be displayed.

        This must be implemented by inheriting classes
        """
        pass

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['map_points'] = self.get_map_points()
        return ctx

    def get_map_points(self):
        """
        Prepare sample data to be passed to the map

        Returns a dict str->str to be turned into json in the template.
        """
        qs = self.get_sample_queryset()
        qs = qs.select_related('dataset')
        fields = ['id', 'sample_name', 'latitude', 'longitude', 'sample_type']

        map_data = []
        for i in qs:
            item = {j: getattr(i, j) for j in fields}

            # add in sample url
            item['sample_url'] = reverse('sample', args=[i.pk])

            # add in dataset info
            item['dataset_url'] = reverse('dataset', args=[i.dataset.pk])
            item['dataset_name'] = str(i.dataset)
            map_data.append(item)

        return map_data


class AbundanceView(ExportMixin, SingleTableView):
    """
    Lists abundance data for a single object of a variable model
    """
    template_name = 'glamr/abundance.html'

    def get_table_class(self):
        if self.model is CompoundAbundance:
            return tables.CompoundAbundanceTable
        elif self.model is FuncAbundance:
            return tables.FunctionAbundanceTable
        elif self.model is TaxonAbundance:
            return tables.TaxonAbundanceTable
        else:
            raise ValueError('unsupported abundance model')

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.object_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.object_model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

        self.model = self.object.abundance.model

    def get_queryset(self):
        try:
            return self.object.abundance.all()
        except AttributeError:
            # (object-)model lacks reverse abundance relation
            raise

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        ctx['object'] = self.object
        ctx['object_model_name'] = self.object_model._meta.model_name

        return ctx


class AbundanceGeneView(ModelTableMixin, SingleTableView):
    """
    Views genes for a sample/something combo

    Can export genes in fasta format
    """
    template_name = 'glamr/abundance_genes.html'
    model = Gene
    exclude = ['id', 'sample']

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.object_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.object_model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

        try:
            self.sample = models.Sample.objects.get(accession=kwargs['sample'])
        except models.Sample.DoesNotExist:
            raise Http404('no such sample')

    def get_queryset(self):
        f = dict(sample=self.sample)
        if self.object_model is FuncRefDBEntry:
            f['besthit__function_refs'] = self.object
        return Gene.objects.filter(**f)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['sample'] = self.sample
        ctx['object'] = self.object
        ctx['object_model_name'] = self.object_model._meta.model_name
        ctx['object_model_name_verbose'] = self.object_model._meta.verbose_name
        return ctx

    # file export methods

    def export_check(self):
        self.export_fasta = 'export-fasta' in self.request.GET
        return self.export_fasta or super().export_check()

    def get_format(self):
        return ('fa/zip', '.fasta.zip', TextRendererZipped)

    def get_filename(self):
        if self.export_fasta:
            return super().get_filename() + '.fasta'
        else:
            return super().get_filename()

    def get_values(self):
        if self.export_fasta:
            return self.get_queryset().to_fasta()
        else:
            return super().get_values()


class BaseDetailView(DetailView):
    template_name = 'glamr/detail.html'
    max_to_many = 16

    field_order = None
    """ a list of field names, setting the order of display, invalid names are
    ignored, fields not listed go last, in the order they are declared in the
    model class """

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object_model_name'] = self.model._meta.model_name
        ctx['object_model_verbose_name'] = self.model._meta.verbose_name
        ctx['details'], ctx['relations'] = self.get_details()
        ctx['external_url'] = self.object.get_external_url()

        return ctx

    def get_details(self):
        details = []
        rel_lists = []
        fields = self.model._meta.get_fields()

        if self.field_order:
            ford = dict(zip(self.field_order, range(len(self.field_order))))
            inf = float('inf')
            fields = sorted(fields, key=lambda x: ford.get(x.name, inf))
            del ford, inf

        for i in fields:
            if i.name == 'id':
                continue

            # some relations (e.g.: 1-1) don't have a verbose name:
            name = getattr(i, 'verbose_name', i.name)

            if i.many_to_many or i.one_to_many:
                model_name = i.related_model._meta.model_name
                try:
                    # trying as m2m relation (other side of declared field)
                    rel_attr = i.get_accessor_name()
                except AttributeError:
                    # this is the m2m field
                    rel_attr = i.name
                qs = getattr(self.object, rel_attr).all()[:self.max_to_many]
                rel_lists.append((name, model_name, qs, i))
                continue

            value = getattr(self.object, i.name, None)
            if value:
                if i.many_to_one or i.one_to_one:  # TODO: test 1-1 fields
                    url = tables.get_record_url(value)
                elif isinstance(i, URLField):
                    url = value
                else:
                    url = None
            else:
                url = None

            if hasattr(i, 'choices') and i.choices:
                value = getattr(self.object, f'get_{i.name}_display')()

            details.append((name, url, value))

        if exturl := self.object.get_external_url():
            details.append(('external URL', exturl, exturl))

        return details, rel_lists


class DatasetView(BaseDetailView):
    model = models.Dataset
    template_name = 'glamr/dataset.html'
    field_order = [
        'reference',
        'sample',
        'bioproject',
        'gold_id',
        'material_type',
        'water_bodies',
        'primers',
        'sequencing_target',
        'sequencing_platform',
        'size_fraction',
        'note',
    ]

    def get_object(self):
        if self.kwargs.get(self.pk_url_kwarg) == 0:
            return models.Dataset.orphans
        return super().get_object()


class DemoFrontPageView(MapMixin, SingleTableView):
    model = models.Dataset
    template_name = 'glamr/demo_frontpage.html'
    table_class = tables.DatasetTable

    filter_class = DatasetFilter
    formhelper_class = DatasetFilterFormHelper
    context_filter_name = 'filter'

    def get_table_data(self):
        data = super().get_table_data()

        self.dataset_ids = data.values_list('id', flat=True)

        orphans = models.Dataset.orphans
        orphans.sample_count = orphans.samples().count()

        # put orphans into first row (if any exist):
        if orphans.sample_count > 0:
            return chain([orphans], data)
        else:
            return data

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.select_related('reference')

        self.filter = self.filter_class(self.request.GET, queryset=qs)
        self.filter.form.helper = self.formhelper_class()

        return self.filter.qs.annotate(sample_count=Count('sample'))

    def get_context_data(self, **ctx):
        # Make the frontpage resilient to database connection issues: Any DB
        # access should be inside a try block catching the first
        # OperationError.  We think covering get_context_data() covers all DB
        # access except those triggered by template rendering.  The template
        # should check for the db_is_good context variable and skip parts of
        # the template that depend on now missing context or parts that trigger
        # further DB access.
        try:
            ctx = self._get_context_data(**ctx)
        except OperationalError as e:
            log.error(f'DB errors on Frontpage: {e.__class__.__name__}: {e}')
            if settings.DEBUG:
                raise
            # In production we expect this to be a DB connection problem, so
            # this is what site visitors are told so they can understand why no
            # data is displayed.
            messages.add_message(
                self.request,
                messages.WARNING,
                'Database connection failure',
            )
            ctx['db_is_good'] = False
        else:
            ctx['db_is_good'] = True

        return ctx

    def _get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['mc_abund'] = TaxonAbundance.objects \
            .filter(taxon__taxname__name='Microcystis') \
            .select_related('sample')[:5]

        ctx[self.context_filter_name] = self.filter

        # Get context for dataset summary
        dataset_counts_df = Dataset.objects.basic_counts()
        dataset_counts_json = dataset_counts_df.reset_index().to_json(orient='records')  # noqa: E501
        dataset_counts_data = json.loads(dataset_counts_json)
        ctx['dataset_counts'] = dataset_counts_data

        ctx['dataset_totalcount'] = \
            Dataset.objects.filter(pk__in=self.dataset_ids).count()
        ctx['sample_totalcount'] = Sample.objects.count()

        # Get context for sample summary
        sample_counts_df = Sample.objects.basic_counts()
        sample_counts_json = sample_counts_df.reset_index().to_json(orient='records')  # noqa: E501
        sample_counts_data = json.loads(sample_counts_json)
        ctx['sample_counts'] = sample_counts_data

        return ctx

    def make_ratios_plot(self):
        # DEPRECATED -- remove?
        imgpath = settings.STATIC_VAR_DIR + '/mappedratios.png'
        ratios = pandas.DataFrame([
            (i.reads_mapped_contigs / i.read_count,
             i.reads_mapped_genes / i.read_count)
            for i in get_sample_model().objects.all()
            if i.contigs_ok and i.genes_ok
        ], columns=['contigs', 'genes'])
        plot = ratios.plot(x='contigs', y='genes', kind='scatter')
        plot.figure.savefig(imgpath)

    def get_sample_queryset(self):
        qs = Sample.objects.filter(dataset__in=self.get_queryset())
        return qs


class ReferenceView(BaseDetailView):
    model = models.Reference


class RecordView(BaseDetailView):
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e


class OverView(SingleTableView):
    template_name = 'glamr/overview.html'
    table_class = tables.OverViewTable

    # lookup from sample to object
    accessor = {
        'compoundentry': 'compoundabundance__compound',
        'funcrefdbentry': 'funcabundance__function',
        'taxon': 'taxonabundance__taxon',
        'compoundname': 'compoundabundance__compound__names',
        'functionname': 'funcabundance__function__names',
    }

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

    def get_table_data(self):
        try:
            a = self.accessor[self.model._meta.model_name]
        except KeyError:
            return []
        else:
            a = 'sample__' + a
            f = {a: self.object}
            qs = models.Dataset.objects.filter(**f)
            qs = qs.annotate(num_samples=Count('sample', distinct=True))

        # add totals (can't do this in above query (cf. django docs order of
        # annotation and filters:
        totals = models.Dataset.objects.filter(pk__in=[i.pk for i in qs])
        totals = totals.annotate(Count('sample'))
        totals = dict(totals.values_list('pk', 'sample__count'))
        for i in qs:
            i.total_samples = totals[i.pk]

        return qs

    def get_table(self):
        table = super().get_table()
        # FIXME: a hack: pass some extra context for column rendering
        table.view_object = self.object
        table.view_object_model_name = self.model._meta.model_name
        return table

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx.update(
            object=self.object,
            object_model_name=self.model._meta.model_name,
            object_model_name_verbose=self.model._meta.verbose_name
        )
        return ctx


class OverViewSamplesView(SingleTableView):
    template_name = 'glamr/overview_samples.html'
    table_class = tables.OverViewSamplesTable

    # lookup from sample to object
    accessor = {
        'compoundentry': 'compoundabundance__compound',
        'funcrefdbentry': 'funcabundance__function',
        'taxon': 'taxonabundance__taxon',
        'compoundname': 'compoundabundance__compound__names',
        'functionname': 'funcabundance__function__names',
    }

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

    def get_table_data(self):
        try:
            a = self.accessor[self.model._meta.model_name]
        except KeyError:
            return []
        else:
            f = {a: self.object}
            qs = models.Sample.objects.filter(**f)
            # distinct: there may be multiple abundances per object and sample
            # e.g. for different roles of a compound
            qs = qs.select_related('dataset').distinct()
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx.update(
            object=self.object,
            object_model_name=self.model._meta.model_name,
            object_model_name_verbose=self.model._meta.verbose_name
        )
        return ctx


class SampleListView(SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = get_sample_model()
    template_name = 'glamr/sample_list.html'
    table_class = tables.SampleTable

    def get_queryset(self):
        pk = self.kwargs['pk']
        if pk == 0:
            self.dataset = models.Dataset.orphans
        else:
            self.dataset = models.Dataset.objects.get(pk=pk)
        return self.dataset.samples()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['dataset'] = str(self.dataset)
        ctx['dataset_id'] = str(self.kwargs['pk'])
        return ctx


class SampleView(BaseDetailView):
    model = get_sample_model()
    template_name = 'glamr/sample_detail.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['sample_id'] = str(self.kwargs['pk'])
        return ctx


class SearchView(TemplateView):
    """ offer a form for advanced search, offer model list """
    template_name = 'glamr/search_init.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['advanced_search'] = True

        # models in alphabetical order
        model_list = [
            (i._meta.model_name, i._meta.verbose_name)
            for i in get_registry().models.values()
        ]
        model_list.sort(key=lambda item: item[1].lower())
        ctx['models'] = model_list

        return ctx


class SearchModelView(EditFilterMixin, TemplateView):
    """ offer model-based searching """
    template_name = 'glamr/search_model.html'
    model = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        return ctx


class SearchHitView(TemplateView):
    template_name = 'glamr/search_hits.html'

    # tri-state indicator if search results hit field data or just reference
    reference_hit_only = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['search_hits'] = self.hits
        ctx['query'] = self.query
        ctx['no_hit_models'] = self.no_hit_models
        ctx['reference_hit_only'] = self.reference_hit_only
        ctx['suggestions'] = self.suggestions
        ctx['field_data_only'] = self.form.cleaned_data.get('field_data_only')
        return ctx

    def get(self, request, *args, **kwargs):
        self.form = AdvancedSearchForm(data=self.request.GET)
        if self.form.is_valid():
            self.query = self.form.cleaned_data['query']
            check_abundance = self.form.cleaned_data.get('field_data_only')
            self.search(check_abundance=check_abundance)
            if check_abundance:
                if self.hits:
                    self.reference_hit_only = False
                else:
                    self.reference_hit_only = True
                    # try again without restriction
                    self.search(check_abundance=False)
        else:
            # invalid form, pretend empty search result
            pass

        if self.hits:
            self.suggestions = None
        else:
            self.suggestions = get_suggestions(self.query)
            if not self.suggestions:
                messages.add_message(
                    request, messages.INFO, 'search: did not find anything'
                )
                return HttpResponseRedirect(reverse('frontpage'))
        return self.render_to_response(self.get_context_data())

    def search(self, check_abundance=False):
        self.hits = []
        self.no_hit_models = []

        qs = models.SearchTerm.objects.filter(term__iexact=self.query)
        if check_abundance:
            qs = qs.filter(has_hit=True)
        qs = qs.order_by('content_type')

        for content_type, grp in groupby(qs, key=lambda x: x.content_type):
            grp = list(grp)
            for i in grp:
                if i.has_hit:
                    have_abundance = True
                    break
            else:
                have_abundance = False
            model = content_type.model_class()
            hits = [
                (i.content_object, str(i.content_object), None)
                for i in grp
            ]
            self.hits.append((
                have_abundance,
                model._meta.verbose_name_plural,
                model._meta.model_name,
                hits,
            ))
        '''


class SampleSearchHitView(MapMixin, TemplateView):
    model = models.Sample
    template_name = 'glamr/search_form_sample_results.html'
    table_class = tables.SampleTable

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['search_results'] = self.results
        ctx['model_name'] = models.Sample._meta.model_name
        ctx['query'] = self.query
        return ctx

    def get(self, request, *args, **kwargs):
        self.form = AdvancedSearchForm(data=self.request.GET)
        if self.form.is_valid():
            self.query = self.form.cleaned_data['query']
            self.search()
        else:
            # invalid form, pretend empty search result
            pass

        return self.render_to_response(self.get_context_data())

    def search(self):
        self.results = []
        qs = Sample.objects.filter(
            Q(geo_loc_name__icontains=self.query) |
            Q(sample_type__icontains=self.query) |
            Q(collection_ts_partial__icontains=self.query) |
            Q(collection_timestamp__icontains=self.query) |
            Q(project_id__icontains=self.query) |
            Q(dataset__scheme__icontains=self.query) |
            Q(env_broad_scale__icontains=self.query) |
            Q(env_local_scale__icontains=self.query) |
            Q(env_medium__icontains=self.query) |
            Q(sample_id__icontains=self.query) |
            Q(dataset__water_bodies__icontains=self.query) |
            Q(dataset__scheme__icontains=self.query) |
            Q(dataset__material_type__icontains=self.query) |
            Q(dataset__reference__short_reference__icontains=self.query) |
            Q(dataset__reference__title__icontains=self.query) |
            Q(dataset__reference__authors__icontains=self.query) |
            Q(sample_name__icontains=self.query)
        ).order_by('sample_name').distinct()

        self.results = qs

    def get_sample_queryset(self):
        return self.results

class DatasetSearchHitView(TemplateView):
    model = models.Dataset
    template_name = 'glamr/search_form_dataset_results.html'
    table_class = tables.DatasetTable

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['search_results'] = self.results
        ctx['model_name'] = models.Dataset._meta.model_name
        ctx['query'] = self.query
        return ctx

    def get(self, request, *args, **kwargs):
        self.results = []
        self.form = AdvancedSearchForm(data=self.request.GET)
        if self.form.is_valid():
            self.query = self.form.cleaned_data['query']
            self.search()
        else:
            # invalid form, pretend empty search result
            pass

        return self.render_to_response(self.get_context_data())

    def search(self):
        qs = Dataset.objects.filter(
            Q(water_bodies__icontains=self.query) |
            Q(material_type__icontains=self.query) |
            Q(reference__short_reference__icontains=self.query) |
            Q(reference__title__icontains=self.query) |
            Q(reference__authors__icontains=self.query) |
            Q(scheme__icontains=self.query) |
            Q(bioproject__icontains=self.query) |
            Q(dataset_id__icontains=self.query) |
            Q(sample__sample_type__icontains=self.query) |
            Q(sample__geo_loc_name__icontains=self.query)
        ).order_by('dataset_id').distinct()
        self.results = qs

    def get_sample_queryset(self):
        return Sample.objects.filter(dataset__in=self.results)


class TableView(BaseFilterMixin, ModelTableMixin, SingleTableView):
    template_name = 'glamr/table.html'

    def get_queryset(self):
        self.conf.q = [self.q]
        return self.conf.get_queryset()

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.conf = TableConfig(self.model)


class ToManyListView(SingleTableView):
    """ view relations belonging to one object """
    template_name = 'glamr/relations_list.html'
    table_class = tables.SingleColumnRelatedTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.obj_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.obj_model.objects.get(pk=kwargs['pk'])
        except self.obj_model.DoesNotExist as e:
            raise Http404('no such record') from e

        try:
            field = self.obj_model._meta.get_field(kwargs['field'])
        except FieldDoesNotExist as e:
            raise Http404('no such field') from e

        if not field.one_to_many and not field.many_to_many:
            raise Http404('field is not *-to_many')

        self.field = field
        self.model = field.related_model

        try:
            self.accessor_name = field.get_accessor_name()
        except AttributeError:
            self.accessor_name = field.name

    def get_queryset(self):
        return getattr(self.object, self.accessor_name).all()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object'] = self.object
        ctx['object_model_name'] = self.obj_model._meta.model_name
        ctx['object_model_name_verbose'] = self.obj_model._meta.verbose_name
        ctx['field'] = self.field
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        return ctx


class ToManyFullListView(ModelTableMixin, ToManyListView):
    """ relations view but with full model-based table """
    template_name = 'glamr/relations_full_list.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        # hide the column for the object
        self.exclude.append(self.field.remote_field.name)
