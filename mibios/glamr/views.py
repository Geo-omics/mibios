from logging import getLogger

from django_tables2 import Column, SingleTableView, TemplateColumn

from django.apps import apps
from django.conf import settings
from django.contrib import messages
from django.core.exceptions import FieldDoesNotExist
from django.db import OperationalError
from django.db.models import Count, Field, Prefetch, URLField
from django.http import Http404, HttpResponse
from django.utils.functional import classproperty
from django.views.generic import DetailView
from django.views.generic.base import TemplateView
from django.views.generic.list import ListView

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
from .forms import QBuilderForm, QLeafEditForm, SearchForm
from .search_fields import SEARCH_FIELDS
from .search_utils import get_suggestions
from .url_utils import fast_reverse

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
        """ Returns True if a file export response is needed """
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
            del col

        for i in self.model._meta.get_fields():
            if i.name in self.exclude:
                continue

            kwargs = {}
            if acc_field and i is acc_field:
                kwargs['linkify'] = \
                    lambda record: tables.get_record_url(record)
            elif not i.many_to_one:
                if hasattr(i, 'unit'):
                    kwargs['verbose_name'] = f'{i.verbose_name} ({i.unit})'
                elif hasattr(i, 'pseudo_unit'):
                    kwargs['verbose_name'] = \
                        f'{i.verbose_name} ({i.pseudo_unit})'
                else:
                    continue
            else:
                # regular FK field
                kwargs['linkify'] = lambda value: tables.get_record_url(value)
            cols.append((i.name, Column(**kwargs)))
        return cols


class MapMixin():
    """
    Mixin for views that display samples on a map
    """
    def get_sample_queryset(self):
        """
        Return a Sample queryset of the samples to be displayed on the map.

        This must be implemented by inheriting classes
        """
        raise NotImplementedError('Inheriting view must implement this method')

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
        qs = qs.values(
            'id', 'sample_name', 'latitude', 'longitude', 'sample_type',
            'dataset_id', 'collection_timestamp', 'sample_id', 'biosample',
        )

        dataset_pks = set((i['dataset_id'] for i in qs))
        datasets = Dataset.objects.filter(pk__in=dataset_pks)
        # str() will access the reference
        datasets = datasets.select_related('reference')
        dataset_name = {i.pk: str(i) for i in datasets}

        map_data = []
        for item in qs:
            # add in sample url
            item['sample_url'] = fast_reverse('sample', args=[item['id']])

            # add in dataset info
            item['dataset_url'] = fast_reverse('dataset', args=[item['dataset_id']])  # noqa:E501
            item['dataset_name'] = dataset_name[item['dataset_id']]
            del item['dataset_id']
            map_data.append(item)

        return map_data


class SearchFormMixin:
    """
    Mixin sufficient to display the simple search form

    Use via including search_form_simple.html in your template.
    """
    model = None  # model class to restrict search to
    ALL_MODELS_URL_PART = 'global'  # url path part indicating global search
    form_class = SearchForm

    _model_class = None

    @classproperty
    def model_class(cls):
        """ a lookup dict to get model classes from name """
        if cls._model_class is None:
            cls._model_class = {
                model_name: apps.get_model(app_label, model_name)
                for app_label, app_models in SEARCH_FIELDS.items()
                for model_name in app_models.keys()
            }
        return cls._model_class

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # add context for search form display:
        ctx['advanced_search'] = False
        ctx['models_and_fields'] = [
            (
                model_name,
                self.model_class[model_name]._meta.verbose_name,
                fields[:7],
            )
            for app, app_data in SEARCH_FIELDS.items()
            for model_name, fields in app_data.items()
        ]

        ctx['search_model'] = self.model
        if self.model:
            ctx['search_radius'] = self.model._meta.model_name
        else:
            ctx['search_radius'] = self.ALL_MODELS_URL_PART
        return ctx


class SearchMixin(SearchFormMixin):
    """
    Mixin to process submitted query, search the database, and display results

    Implementing classes need to call the search() method.
    """
    _verbose_field_name = None

    @classproperty
    def verbose_field_name(cls):
        """
        map field names to field's verbose_name

        Returns a dict-of-dict mapping model names to dicts field
        names->verbose field name.
        """
        if cls._verbose_field_name is None:
            cls._verbose_field_name = {}
            for app_label, app_models in SEARCH_FIELDS.items():
                for model_name, fields in app_models.items():
                    model = cls.model_class[model_name]
                    verbose = {}
                    for i in fields:  # fields is list of field names
                        verbose[i] = model._meta.get_field(i).verbose_name
                    cls._verbose_field_name[model_name] = verbose
        return cls._verbose_field_name

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            model = kwargs['model']
        except KeyError as e:
            raise ValueError('expect model name passed from url conf') from e

        if model == self.ALL_MODELS_URL_PART:
            self.model = None
        else:
            try:
                self.model = self.model_class[model]
            except KeyError as e:
                raise Http404('bad model name / is not searchable') from e

        self.query = None
        self.check_abundance = False
        self.search_result = {}
        self.suggestions = []

    def process_search_form(self):
        """
        Process the user input

        Called from search()
        """
        self.results = {}
        form = self.form_class(data=self.request.GET)
        if form.is_valid():
            self.query = form.cleaned_data['query']
            self.check_abundance = form.cleaned_data.get('field_data_only', False)  # noqa: E501
        else:
            # invalid form, pretend empty search result
            log.debug(f'form errors: {form.errors=} {self.request.GET=}')
            pass

    def search(self):
        """
        Do the full-text search

        Depending on the search's success, this methods sets the view's
        search_result and suggestions attributes.
        """
        self.process_search_form()
        if not self.query:
            return
        self.search_result = models.Searchable.objects.search(
            query=self.query,
            models=[self.model._meta.model_name] if self.model else [],
            abundance=self.check_abundance,
        )
        if not self.search_result:
            self.suggestions = get_suggestions(self.query)
            if not self.suggestions:
                messages.add_message(
                    self.request, messages.INFO,
                    'search: did not find anything'
                )

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # add context to display search results:
        ctx['query'] = self.query
        ctx['verbose_field_name'] = self.verbose_field_name
        if self.search_result and self.model is None:
            # issue results statistics unless a single model was searched
            ctx['result_stats'] = [
                (model, sum((len(items) for items in model_results.values())))
                for model, model_results in self.search_result.items()
            ]
        else:
            ctx['result_stats'] = None
        return ctx


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

    hidden_fields = set()
    """ a list of field names of fields that should not appear in the view """

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
            if i.name == 'id' or i.name in self.hidden_fields:
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

            if value:
                unit = getattr(i, 'unit', None)
            else:
                unit = None

            pseudo_unit = getattr(i, 'pseudo_unit', None)

            details.append((name, pseudo_unit, url, value, unit))

        if exturl := self.object.get_external_url():
            details.append(('external URL', None, exturl, exturl, None))

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


class FrontPageView(SearchFormMixin, MapMixin, SingleTableView):
    model = models.Dataset
    template_name = 'glamr/frontpage.html'
    table_class = tables.DatasetTable

    filter_class = DatasetFilter
    formhelper_class = DatasetFilterFormHelper
    context_filter_name = 'filter'

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(private=False)
        qs = qs.select_related('reference')

        # to get sample type in table
        qs = qs.prefetch_related(Prefetch(
            'sample_set',
            queryset=Sample.objects.only('dataset_id', 'sample_type'),
        ))

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
                'Database connection failure: almost nothing will work, sorry',
            )
            ctx['db_is_good'] = False
            ctx['search_radius'] = 'global'
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

        ctx['dataset_totalcount'] = Dataset.objects.count()
        ctx['filtered_dataset_totalcount'] = self.filter.qs.count()
        ctx['sample_totalcount'] = Sample.objects.count()

        # Get context for sample summary
        sample_counts_df = Sample.objects.basic_counts()
        sample_counts_json = sample_counts_df.reset_index().to_json(orient='records')  # noqa: E501
        sample_counts_data = json.loads(sample_counts_json)
        ctx['sample_counts'] = sample_counts_data

        return ctx

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


class SampleListView(ExportMixin, SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = get_sample_model()
    template_name = 'glamr/sample_list.html'
    table_class = tables.SampleTable

    def get_queryset(self):
        pk = self.kwargs['pk']
        self.dataset = models.Dataset.objects.get(pk=pk)
        return self.dataset.samples()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['dataset'] = self.dataset
        return ctx


class SampleView(BaseDetailView):
    model = get_sample_model()
    template_name = 'glamr/sample_detail.html'
    hidden_fields = [
        'meta_data_loaded',
    ]

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['sample_id'] = str(self.kwargs['pk'])
        return ctx


class SearchView(TemplateView):
    """ offer a form for advanced search, offer model list """
    template_name = 'glamr/search_init.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['advanced_search'] = False
        m_and_f = []
        for app, app_data in SEARCH_FIELDS.items():
            app = apps.get_app_config(app)
            for model_name, fields in app_data.items():
                m = app.get_model(model_name)
                m_and_f.append((model_name, m._meta.verbose_name, fields[:7]))
        ctx['models_and_fields'] = m_and_f

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


class ResultListView(SearchMixin, MapMixin, ListView):
    template_name = 'glamr/result_list.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['suggestions'] = self.suggestions
        return ctx

    def result_to_list(self, result):
        """
        Helper to turn SearchMixin.search_result into a ListView.object_list
        """
        res = []
        for model, items_per_field in result.items():
            model_name = model._meta.model_name
            items = {
                pk: (field, text)
                for field, items in items_per_field.items()
                for text, pk in items
            }
            objs = model.objects.in_bulk(items.keys())
            is_first = True  # True for the first item of each model
            for pk, (field, text) in items.items():
                if pk not in objs:
                    # search index out-of-sync
                    continue
                field = self.verbose_field_name[model_name][field]
                res.append((is_first, model_name, objs[pk], field, text))
                is_first = False
        return res

    def get_queryset(self):
        self.search()
        if self.search_result:
            return self.result_to_list(self.search_result)
        else:
            return []

    def get_sample_queryset(self):

        sample_pks = set()
        for _, result_items in self.search_result.get(Sample, {}).items():
            for _, pk in result_items:
                sample_pks.add(pk)
        for _, result_items in self.search_result.get(Dataset, {}).items():
            dataset_pks = [pk for _, pk in result_items]
            sample_pks.update(
                Sample.objects.filter(dataset__pk__in=dataset_pks)
                .values_list('pk', flat=True)
            )
        return Sample.objects.filter(pk__in=sample_pks)


class FilteredListView(SearchFormMixin, MapMixin, ModelTableMixin,
                       SingleTableView):
    template_name = 'glamr/filter_list.html'

    def setup(self, request, *args, model=None, **kwargs):
        super().setup(request, *args, **kwargs)
        # support searchable models only for now
        try:
            self.model = self.model_class[model]
        except KeyError as e:
            raise Http404('model not supported in this view') from e
        self.conf = TableConfig(self.model)

    def set_filter(self):
        """ set filter from GET querystring """
        f = {}
        for key, val in self.request.GET.items():
            fname = key.split('__')[0]
            try:
                self.model._meta.get_field(fname)
            except FieldDoesNotExist:
                continue
            f[key] = val
        self.conf.filter = f

    def get_queryset(self):
        self.set_filter()
        return self.conf.get_queryset()

    def get_sample_queryset(self):
        if self.model is Sample:
            return self.get_queryset()
        if self.model is Dataset:
            return Sample.objects.filter(dataset__in=self.get_queryset())
        else:
            # TODO
            return Sample.objects.none()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        self.set_filter()

        ctx['filter_items'] = [
            (k.replace('__', ' -> '), v)
            for k, v in self.conf.filter.items()
        ]

        if self.model is Sample:
            ctx['filter_model'] = "sample"
        elif self.model is Dataset:
            ctx['filter_model'] = "dataset"
        else:
            ctx['filter_model'] = "generic"

        return ctx
