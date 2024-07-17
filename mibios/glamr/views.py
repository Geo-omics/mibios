from itertools import groupby
from logging import getLogger
import pprint
import re

from django_tables2 import (
    Column, LazyPaginator, SingleTableView, TemplateColumn, table_factory,
)

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import FieldDoesNotExist
from django.db import OperationalError, connection
from django.db.models import Count, Field, Prefetch, URLField
from django.http import Http404, HttpResponse
from django.urls import reverse
from django.utils.functional import cached_property, classproperty
from django.utils.html import format_html
from django.views.decorators.cache import cache_page
from django.views.generic import DetailView
from django.views.generic.base import TemplateView, View
from django.views.generic.list import ListView

from mibios import get_registry
from mibios.data import DataConfig, TableConfig
from mibios.glamr.filters import (
    DatasetFilter, UniRef90Filter, UniRef100Filter,
    filter_registry
)
from mibios.glamr.forms import DatasetFilterFormHelper
from mibios.glamr.models import Sample, Dataset, pg_class, dbstat
from mibios.models import Q
from mibios.views import ExportBaseMixin, TextRendererZipped, VersionInfoMixin
from mibios.omics import get_sample_model
from mibios.omics.models import (
    CompoundAbundance, FuncAbundance, ReadAbundance, TaxonAbundance,
    SampleTracking,
)
from mibios.ncbi_taxonomy.models import TaxNode
from mibios.umrad.models import FuncRefDBEntry, UniRef100
from mibios.umrad.utils import DefaultDict
from mibios.omics.models import File, Gene
from mibios.omics.views import RequiredSettingsMixin
from . import models, tables, GREAT_LAKES
from .forms import QBuilderForm, QLeafEditForm, SearchForm
from .search_fields import ADVANCED_SEARCH_MODELS, search_fields
from .search_utils import get_suggestions
from .url_utils import fast_reverse
from .utils import estimate_row_totals, get_record_url


log = getLogger(__name__)


class BaseMixin(VersionInfoMixin):
    def dispatch(self, request, *args, cache=True, **kwargs):
        if cache:
            return cache_page(300)(super().dispatch)(request, *args, **kwargs)
        else:
            return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['show_internal_nav'] = settings.INTERNAL_DEPLOYMENT
        try:
            # pick up django_table2 table name
            ctx['version_info']['table'] = ctx['table'].__class__.__name__
        except Exception:
            ctx['version_info']['table'] = 'none'
        return ctx


class ExportMixin(ExportBaseMixin):
    """
    Allow data download via query string parameter

    Should be used together with a django_tables2 table view.
    Also requires BaseMixin to pass around the cache option.
    """
    export_query_param = 'export'
    export_options = None

    exclude = ['id']
    """ Which fields to exclude """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instances get their own lists as they may modify them:
        self.exclude = list(self.exclude)

    def get_filename(self, option=''):
        parts = []
        if hasattr(self, 'object_model') and self.object_model:
            parts.append(self.object_model._meta.verbose_name)
        if hasattr(self, 'object') and self.object:
            parts.append(self.object)
        if hasattr(self, 'model') and self.model:
            parts.append(self.model._meta.verbose_name_plural)

        if option:
            try:
                field = self.model._meta.get_field(option)
            except Exception:
                parts.append(option)
            else:
                parts.append(field.related_name or field.name)

        if parts:
            value = '-'.join((str(i) for i in parts if i))
        if not value:
            value = self.__class__.__name__.lower() + '-export'

        value = value.replace('_', '-')
        value = value.replace(' ', '-')
        return value

    def dispatch(self, request, *args, cache=False, **kwargs):
        # do not cache data export
        cache = cache or self.export_check() is None
        return super().dispatch(request, *args, cache=cache, **kwargs)

    def get(self, request, *args, **kwargs):
        export_opt = self.export_check()
        if export_opt is None:
            self.populate_export_options()
            return super().get(request, *args, **kwargs)

        return self.export_response(export_opt)

    def export_check(self):
        """
        Evaluate the GET query string for the export parameter

        Returns the export option (a str) if any, or None if not export reponse
        was requested, meaning the data will then be displayed normally as
        HTML.  Returns the empty string to export the normal table.  Otherwise
        it is an accessor to a relation.
        """
        export_opt = self.request.GET.get(self.export_query_param)
        # With GET.get() we get None if the key does not appear in the query
        # string.  It's an empty string if the key is there but without a
        # value, otherwise we get the value, and the last such value should
        # there be multiple export params.
        if export_opt not in [None, '']:
            # TODO: check for valid parameter?
            ...

        return export_opt

    def export_response(self, option):
        """ generate file download response """
        name, suffix, renderer_class = self.get_format()

        response = HttpResponse(content_type=renderer_class.content_type)
        filename = self.get_filename(option) + suffix
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        renderer = renderer_class(response, filename=filename)
        renderer.render(self.get_values(option))

        return response

    def get_values(self, option):
        """
        Collect data to be exported
        """
        if option == '':
            # export current table
            if hasattr(self, 'get_table'):
                return self.get_table(**self.get_table_kwargs()).as_values()
            else:
                raise Http404('export not implemented')

        try:
            # try exporting related data
            field = self.model._meta.get_field(option)
        except FieldDoesNotExist:
            pass
        else:
            f = {field.remote_field.name + '__in': self.get_queryset()}
            qs = field.related_model.objects.filter(**f)
            # return qs.values_list()
            try:
                # for use with ModelTableMixin:
                tab_cls = self.TABLE_CLASSES[field.related_model]
            except (AttributeError, KeyError):
                raise
                tab_cls = table_factory(field.related_model, tables.Table)
            return tab_cls(data=qs).as_values()

        raise Http404('the given export option is not implemented')

    def populate_export_options(self):
        """ Automatically fill in export options """
        if hasattr(self, 'get_queryset'):
            # if we have a queryset, we want to offer downloading it
            self.add_export_option('')

    def add_export_option(self, option):
        disable_link = False
        if self.export_options is None:
            self.export_options = []

        if option == '':
            # default export, the view's model
            link_txt = self.model._meta.verbose_name_plural
            link_txt += ' (this table)'
        else:
            try:
                field = self.model._meta.get_field(option)
            except FieldDoesNotExist:
                raise ValueError(f'unsupported export option value: {option}')
            if not field.one_to_many or field.many_to_many:
                raise ValueError(f'field {field} is not *-to-many')
            link_txt = field.related_name \
                or field.related_model._meta.verbose_name_plural

            if not settings.INTERNAL_DEPLOYMENT:
                # currently impractical, use too much resources
                if link_txt == 'functional_abundance':
                    disable_link = True

        qstr = self.request.GET.copy()
        qstr[self.export_query_param] = option
        url = f'?{qstr.urlencode()}'
        if disable_link:
            url = None
        self.export_options.append((url, link_txt))

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['export_options'] = self.export_options
        return ctx


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


class FilterMixin:
    """
    Mixin for django-filter
    """
    def get(self, request, *args, **kwargs):
        try:
            fclass = filter_registry.from_get(request.GET)
        except LookupError:
            # GET qstr does not match filter signature
            try:
                fclass = filter_registry.by_model[self.model]
            except KeyError:
                fclass = None
        else:
            if self.model is not fclass._meta.model:
                raise Http404('model is incompatible with filter code ')

        self.filter_class = fclass
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        qs = super().get_queryset()
        if self.filter_class:
            self.filter = self.filter_class(self.request.GET, qs)
            qs = self.filter.qs
        else:
            self.filter = None
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['filter'] = self.filter
        if self.filter:
            ctx['filter_items'] = self.filter.for_display()
        else:
            ctx['filter_items'] = []
        if 'version_info' in ctx:
            ctx['version_info']['filter'] = self.filter.__class__.__name__
        return ctx


_estimated_row_totals_cache = {}


def get_row_totals_estimate(model):
    if model not in _estimated_row_totals_cache:
        _estimated_row_totals_cache[model] = estimate_row_totals(model)
    return _estimated_row_totals_cache[model]


class ModelTableMixin(ExportMixin):
    """
    Mixin for SingleTableView

    To view a table with model-specific customization if available or fall-back
    to generic table.

    Improves columns for relation fields.  The inheriting view must set
    self.model
    """
    model = None  # model needs to be set by inheriting class

    table_class = None
    """ The table class can be set by an implementing class, if it remains
    None, then the class will be picked from the TABLE_CLASSES dictionary or as
    a last resort the factory will create a class based on the model.
    """

    LAZY_PAGINATION_THRESHOLD = 10000

    TABLE_CLASSES = {
        Dataset: tables.DatasetTable,
        Sample: tables.SampleTable,
        TaxonAbundance: tables.TaxonAbundanceTable,
        ReadAbundance: tables.ReadAbundanceTable,
        TaxNode: tables.TaxNodeTable,
        File: tables.FileTable,
    }

    EXTRA_EXPORT_OPTIONS = {
        Sample: ['taxonabundance', 'functional_abundance'],
    }

    def setup(self, request, *args, model=None, **kwargs):
        """
        setup: Sets model from url kwarg, if possible.  If this default
        behaviour is not correct for an inheriting view, then override this, so
        call super().setup() and then set self.model as desired.
        """
        super().setup(request, *args, **kwargs)
        if model:
            try:
                self.model = get_registry().models[model]
            except KeyError as e:
                raise Http404(f'no such model: {e}') from e

        try:
            est = get_row_totals_estimate(self.model)
        except Exception as e:
            log.error(f'estimating row totals failed: {e}')
            self.paginator_class = LazyPaginator
            raise
        else:
            if est > self.LAZY_PAGINATION_THRESHOLD:
                self.paginator_class = LazyPaginator

    def get_table_kwargs(self):
        kw = super().get_table_kwargs()
        kw['exclude'] = list(self.exclude)
        if self.model not in self.TABLE_CLASSES:
            kw['extra_columns'] = self._get_improved_columns()
        kw['view'] = self
        return kw

    def get_table_class(self):
        if self.table_class:
            return self.table_class

        # Some model have dedicated tables, everything else gets a table from
        # the factory at SingleTableMixin
        return self.TABLE_CLASSES.get(
            self.model,
            table_factory(self.model, tables.Table),
        )

    def get_queryset(self):
        """
        Also run model-specific methods on the queryset

        Anything non-generic that a custom table needs should go in here.
        Otherwise model-agnostic views, that implement this mixin, should call
        this method in get_queryset().
        """
        qs = super().get_queryset()
        if self.model is Dataset:
            qs = qs.annotate(sample_count=Count('sample', distinct=True))
        return qs

    def _get_improved_columns(self):
        """ make replacements to linkify FK + accession columns """
        cols = []
        try:
            acc_field = self.model.get_accession_field_single()
        except (RuntimeError, LookupError):
            acc_field = None
            col = TemplateColumn(
                '{%load glamr_extras%}[<a href="{% record_url record %}">'
                + self.model._meta.model_name
                + '/{{ record.pk }}</a>]',
                order_by='pk',
            )
            cols.append(('record links', col))
            del col

        for i in self.model._meta.get_fields():
            if i.name in self.exclude:
                continue

            kwargs = {}
            if acc_field and i is acc_field:
                kwargs['linkify'] = tables.linkify_record
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
                kwargs['linkify'] = tables.linkify_value
            cols.append((i.name, Column(**kwargs)))
        return cols

    def get_context_data(self, **ctx):
        # add export opts before ExportMixin makes context data:
        for i in self.EXTRA_EXPORT_OPTIONS.get(self.model, []):
            self.add_export_option(i)
        if hasattr(self.object_list, 'extra_context'):
            ctx.update(self.object_list.extra_context())
        ctx = super().get_context_data(**ctx)
        ctx['model_name'] = self.model._meta.model_name
        return ctx


class MapMixin():
    """
    Mixin for views that display samples on a map
    """
    def get_sample_queryset(self):
        """
        Return a Sample queryset of the samples to be displayed on the map.
        """
        if self.model is Sample:
            return self.get_queryset()
        elif self.model is Dataset:
            if hasattr(self, 'conf') and self.conf is not None:
                return self.conf.shift('sample', reverse=True).get_queryset()
            else:
                return Sample.objects.filter(dataset__in=self.get_queryset())
        else:
            return Sample.objects.none()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['map_points'] = self.get_map_points()
        ctx['fit_map_to_points'] = True
        return ctx

    def get_map_points(self):
        """
        Prepare sample data to be passed to the map

        Returns a dict str->str to be turned into json in the template.
        """
        qs = self.get_sample_queryset()
        qs = qs.exclude(longitude='')
        qs = qs.exclude(latitude='')
        qs = qs.select_related('dataset')
        qs = qs.values(
            'id', 'sample_name', 'latitude', 'longitude', 'sample_type',
            'dataset_id', 'collection_timestamp', 'sample_id', 'biosample',
        )
        qs = qs.order_by('longitude', 'latitude')

        dataset_pks = set((i['dataset_id'] for i in qs))
        datasets = Dataset.objects.filter(pk__in=dataset_pks)
        # str() will access the reference
        datasets = datasets.select_related('primary_ref')
        dataset_name = {i.pk: str(i) for i in datasets}
        dataset_no = {i.pk: i.get_set_no() for i in datasets}

        base_cnf = DataConfig(Sample)
        map_data = []
        by_coords = groupby(qs, key=lambda x: (x['longitude'], x['latitude']))
        for coords, grp in by_coords:
            grp = list(grp)

            # take only the first sample at these coordinates
            item = grp[0]

            item['sample_url'] = fast_reverse(
                'sample', args=[item['sample_id'].removeprefix('samp_')],
            )
            item['dataset_url'] = fast_reverse(
                'dataset', args=[dataset_no[item['dataset_id']]],
            )
            item['dataset_name'] = dataset_name[item['dataset_id']]
            del item['dataset_id']

            # types_at_location: construct a CSS selector prefix for the map
            # marker/icon.  The map javascript will add '-icon'.  Assumes that
            # any possible selector is defined in the loaded style sheet.  If
            # no sample in the group has a sample_type then we put in an empty
            # string, resulting in a (hopefully) invalid selector in which case
            # no marker will be displayed.
            stypes = set((i['sample_type'] for i in grp if i['sample_type']))
            item['types_at_location'] = '-'.join(sorted(stypes))

            if len(grp) > 1:
                # Add a link to the other samples at these coordinates.  We try
                # to keep existing filters in place but if that fails we link
                # to all samples at this location.
                if hasattr(self, 'conf') and self.conf is not None:
                    if self.conf.model is Dataset:
                        cnf = self.conf.shift('sample', reverse=True)
                    elif self.conf.model is Sample:
                        cnf = self.conf
                    else:
                        cnf = base_cnf
                else:
                    cnf = base_cnf
                cnf = cnf.add_filter(
                    longitude=item['longitude'],
                    latitude=item['latitude'],
                )
                item['others'] = format_html(
                    '<br>and {} other '
                    '<a href="{}?{}">samples at these coordinates</a>',
                    len(grp) - 1,
                    reverse('filter_result', kwargs=dict(model='sample')),
                    cnf.url_query(),
                )

            map_data.append(item)

        return map_data


class SearchFormMixin:
    """
    Mixin sufficient to display the simple search form

    Use via including full_text_search_form.html in your template.
    """

    search_model = None
    """ Model class to restrict search to.  Implementing classes may override
    this.  It may get the special value ANY_MODEL, which is distinct from None.
    None indicates that the value of the model attribute shall be used.
    """

    model = None
    """ Typically set by the implementing class or via request.  Here only used
    to set the search_model if search_model is not set otherwise. """

    ANY_MODEL = object()
    ANY_MODEL_URL_PART = 'global'  # url path part indicating global search
    form_class = SearchForm

    _model_class = None

    @classproperty
    def model_class(cls):
        """ a lookup dict to get model classes from name """
        if cls._model_class is None:
            cls._model_class = {
                i._meta.model_name: i
                for i in search_fields.keys()
            }
        return cls._model_class

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # add context for search form display:
        ctx['advanced_search'] = False
        ctx['models_and_fields'] = [
            (
                model._meta.model_name,
                model._meta.verbose_name_plural,
                fields[:7] + (
                    ['...'] if len(fields) > 7 else []
                ),
            )
            for model, fields in search_fields.items()
        ]

        search_model = self.search_model or self.model
        if search_model in [self.ANY_MODEL, None]:
            ctx['search_model'] = None
            ctx['search_model_name'] = self.ANY_MODEL_URL_PART
        else:
            ctx['search_model'] = search_model
            ctx['search_model_name'] = search_model._meta.model_name
        return ctx


class SearchMixin(SearchFormMixin):
    """
    Mixin to process submitted query, search the database, and display results

    Implementing classes need to call the search() method.
    """
    SHOW_MORE_INCREMENT = 25
    DEFAULT_LIMIT = 25

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            model_name = kwargs['model']
        except KeyError as e:
            raise ValueError('expect model name passed from url conf') from e

        if model_name == self.ANY_MODEL_URL_PART:
            self.model = None
            self.search_model = self.ANY_MODEL
        else:
            try:
                self.model = self.model_class[model_name]
            except KeyError as e:
                raise Http404('bad model name / is not searchable') from e

        self.query = None
        self.soft_limit = self.DEFAULT_LIMIT
        self.real_query = None
        self.check_abundance = False
        self.search_result = {}
        self.suggestions = []
        self.did_fallback_search = False

    def process_search_form(self):
        """
        Process the user input

        Called from search()
        """
        self.results = {}
        form = self.form_class(data=self.request.GET)
        if form.is_valid():
            self.query = form.cleaned_data['query']
            self.soft_limit = form.cleaned_data['limit'] or self.DEFAULT_LIMIT
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

        Returns a dict.  Sets self.suggestions as a side-effect.
        """
        self.process_search_form()
        if not self.query:
            return {}

        search_kwargs = {
            'query': self.query,
            'models': [self.model._meta.model_name] if self.model else [],
            'abundance': self.check_abundance,
            'soft_limit': max(self.soft_limit, self.DEFAULT_LIMIT),
        }

        # first search
        search_result = models.Searchable.objects.search(**search_kwargs)

        if not search_result:
            # get and process spelling suggestions
            suggs = get_suggestions(self.query)
            orig_phrase = suggs.keys()
            suggestions = {}  # these are to be displayed
            urlpath = reverse('search_result', kwargs=self.kwargs)
            for idx, (word, match_list) in enumerate(suggs.items()):
                if match_list is None:
                    # word is good
                    continue
                else:
                    # word is misspelled
                    suggestions[word] = []
                    for i in match_list:
                        alt_query = list(orig_phrase)
                        alt_query[idx] = i
                        alt_query = '+'.join(alt_query)
                        url = f'{urlpath}?query={alt_query}'
                        suggestions[word].append((i, url))

            self.suggestions = suggestions

            # fallback search
            search_result = models.Searchable.objects.fallback_search(**search_kwargs)  # noqa:E501
            self.did_fallback_search = True

        if not search_result and not self.suggestions:
            messages.add_message(
                self.request, messages.INFO,
                'search: did not find anything'
            )

        return search_result

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # add context to display search results:
        ctx['query'] = self.query
        ctx['did_fallback_search'] = self.did_fallback_search
        ctx['total_count'] = self.search_result.get_total_hit_count()
        ctx['total_count_at_limit'] = self.search_result.reached_hard_limit()
        if self.search_result and self.model is None:
            # issue results statistics unless a single model was searched
            ctx['result_stats'] = self.search_result.get_stats()
        else:
            ctx['result_stats'] = None
        return ctx


class AboutView(BaseMixin, DetailView):
    template_name = 'glamr/about.html'
    model = models.AboutInfo

    def get_object(self):
        # last() may also return None so provide a blank instance useful for
        # development.
        self.object = self.get_queryset().order_by('pk').last() or self.model()
        return self.object

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['credits'] = self.get_credits()
        ctx['have_history'] = self.model.objects.count() >= 2
        return ctx

    def get_credits(self):
        """ Compile credits data """
        data = []
        naked_version_pat = re.compile(r'[0-9.]+')

        if self.object.pk is None:
            # got a blank instance, rf. get_object(), the m2m credits
            # relationship won't work, so shortcut here and return an empty
            # list
            return data

        credits = self.object.credits.order_by('group', 'name').all()
        for _, objs in groupby(credits, key=lambda x: x.group):
            credit_data = []
            for i in objs:
                if i.version:
                    if naked_version_pat.match(i.version):
                        version = f'v{i.version}'
                    else:
                        version = i.version
                else:
                    version = i.version
                version_info = \
                    ' '.join((str(i) for i in (version, i.date, i.time) if i))

                url = None
                source_url = None
                paper_url = None
                if i.website:
                    url = i.website
                    source_url = i.source_code
                    paper_url = i.paper
                elif i.source_code:
                    url = i.source_code
                    paper_url = i.paper
                elif i.paper:
                    url = paper_url

                credit_data.append((
                    i.name,
                    url,
                    version_info,
                    source_url,
                    paper_url,
                    i.comment,
                ))
            # get human-readable group name from last object
            data.append((i.get_group_display(), credit_data))
        return data


class AboutHistoryView(BaseMixin, SingleTableView):
    template_name = 'glamr/about_history.html'
    model = models.AboutInfo
    table_class = tables.AboutHistoryTable


class AbundanceView(MapMixin, ModelTableMixin, BaseMixin, SingleTableView):
    """
    Lists abundance data for a single object of certain models
    """
    template_name = 'glamr/abundance.html'

    # view attributes set by setup:
    VIEW_ATTRS = {
        'taxnode': {
            'model': TaxonAbundance,
            'table_class': tables.TaxonAbundanceTable,
            'sample_filter_key': 'taxonabundance__taxon',
        },
        'uniref100': {
            'model': ReadAbundance,
            'table_class': tables.ReadAbundanceTable,
            'sample_filter_key': 'functional_abundance__ref',
        },
        'funcrefdbentry': {
            'model': FuncAbundance,
            'table_class': tables.FunctionAbundanceTable,
            'sample_filter_key': 'funcabundance__function',
        },
        'compoundrecord': {
            'model': CompoundAbundance,
            'table_class': tables.CompoundAbundanceTable,
            'sample_filter_key': 'compoundabundance__compound',
        },
    }

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        obj_model_name = kwargs['model']
        try:
            view_attrs = self.VIEW_ATTRS[obj_model_name]
        except KeyError as e:
            raise Http404(f'unsupported model: {e}') from e

        for key, value in view_attrs.items():
            setattr(self, key, value)

        try:
            self.object_model = get_registry().models[obj_model_name]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.object_model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

        self.model = self.object.abundance.model

    def get_queryset(self):
        try:
            qs = self.object.abundance.all()
        except AttributeError:
            # (object-)model lacks reverse abundance relation
            raise

        qs = qs.filter(sample__dataset__private=False)
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        ctx['object'] = self.object
        ctx['object_model_name'] = self.object_model._meta.model_name
        return ctx

    def get_sample_queryset(self):
        f = {self.sample_filter_key: self.object}
        return Sample.objects.filter(**f)


class AbundanceGeneView(ModelTableMixin, BaseMixin, SingleTableView):
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


class ContactView(BaseMixin, TemplateView):
    template_name = 'glamr/contact.html'


class DBInfoView(RequiredSettingsMixin, BaseMixin, SingleTableView):
    required_settings = 'INTERNAL_DEPLOYMENT'
    template_name = 'glamr/dbinfo.html'
    model = None  # set by setup()
    table_class = tables.DBInfoTable
    table_pagination = False

    table_prefixes = (
        'django_',
        'glamr_',
        'omics_',
        'ncbi_taxonomy_',
        'umrad_',
    )

    MODEL = {
        'postgresql': pg_class,
        'sqlite': dbstat,
    }

    def setup(self, *args, **kwargs):
        try:
            self.model = self.MODEL[connection.vendor]
        except KeyError as e:
            raise Http404(f'unsupported db vendor: {e}')
        return super().setup(*args, **kwargs)

    def get_queryset(self):
        qs = super().get_queryset()
        q = Q()
        for pref in self.table_prefixes:
            q = q | Q(name__startswith=pref)
        return qs.filter(q)


class RecordView(BaseMixin, DetailView):
    """
    View details of a single object of any model
    """
    template_name = 'glamr/detail.html'
    max_to_many = 16

    fields = None
    """ a list of field names, setting the order of display, invalid names are
    ignored, fields not listed go last, in the order they are declared in the
    model class """
    # FIXME: doc for fields is not accurate

    related = None
    """ *-to-many fields for which to show objects """

    exclude = []
    """ a list of field names of fields that should not appear in the view """

    OTHERS = object()

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if self.model is None:
            # generic case, need to get the model from URL:
            if 'model' not in kwargs:
                raise ValueError('no model specified by class nor by url')

            try:
                self.model = get_registry().models[kwargs['model']]
            except KeyError as e:
                raise Http404(f'no such model: {e}') from e

    def get_object_lookups(self):
        """
        Compile Queryset filter from kwargs

        This default implementation only supports lookup by primary key via the
        'key' kwarg.  Overwrite this to support other kinds of lookup.
        """
        return dict(pk=self.kwargs['key'])

    def get_object(self, lookups=None, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()

        lookups = self.get_object_lookups()
        try:
            obj = queryset.filter(**lookups).get()
        except self.model.DoesNotExist:
            raise Http404(f'no {self.model} matching query {lookups=}')

        return obj

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object_model_name'] = self.model._meta.model_name
        ctx['object_model_verbose_name'] = self.model._meta.verbose_name
        ctx['details'] = self.get_details()
        ctx['relations'] = self.get_relations()
        ctx['external_url'] = self.object.get_external_url()
        ctx['has_abundance'] = (
            self.model._meta.model_name in AbundanceView.VIEW_ATTRS
        )

        return ctx

    def get_ordered_fields(self):
        """
        Helper to make a correctly ordered list of fields
        """
        fields = []
        if self.fields:
            for i in self.fields:
                if i in self.exclude:
                    continue

                if i is RecordView.OTHERS:
                    continue

                try:
                    fields.append(self.model._meta.get_field(i))
                except FieldDoesNotExist:
                    if hasattr(self.model, i):
                        fields.append(i)
                    else:
                        raise AttributeError(
                            f'{self.model} has no field or attribute "{i}"'
                        )

            try:
                others_pos = self.fields.index(RecordView.OTHERS)
            except ValueError:
                # no others
                others_pos = None
        else:
            # self.fields not set, auto-add below
            others_pos = -1

        if others_pos is not None:
            others = []
            for i in self.model._meta.get_fields():
                if i in fields:
                    continue
                if i.name == 'id' or i.name in self.exclude:
                    continue
                if i.name in self.model.get_internal_fields():
                    continue
                others.append(i)
            fields = fields[:others_pos] + others + fields[others_pos + 1:]
        return fields

    def get_relations(self):
        """
        Get details on *-to-many fields
        """
        fields = []
        if self.related is None:
            # the default, display all *-to-many fields
            for i in self.model._meta.get_fields():
                if i.one_to_many or i.many_to_many:
                    fields.append(i)
        elif self.related:
            # only diplay specified fields
            for i in self.related:
                fields.append(self.model._meta.get_field(i))
        else:
            # self.related is set empty  ==>  show nothing
            return []

        data = []
        for i in fields:
            if i.related_model is models.Searchable:
                continue
            model_name = i.related_model._meta.model_name
            name = getattr(i, 'verbose_name_plural', i.name)
            try:
                # trying as m2m relation (other side of declared field)
                rel_attr = i.get_accessor_name()
            except AttributeError:
                # this is the m2m field
                rel_attr = i.name
            qs = getattr(self.object, rel_attr).all()[:self.max_to_many]
            data.append((name, model_name, qs, i))

        return data

    def get_detail_for_field(self, field):
        """ get the 4-tuple detail data for a field """
        f = field
        if f.one_to_many or f.many_to_many:
            name = f.related_model._meta.verbose_name_plural
            # make value the count of related objects!
            try:
                # trying as m2m relation (other side of declared field)
                rel_attr = f.get_accessor_name()
            except AttributeError:
                # this is the m2m field
                rel_attr = f.name
            value = getattr(self.object, rel_attr).count()
            is_blank = (value == 0)  # Let's not show zeros
        elif f.one_to_one:
            # 1-1 fields don't have a verbose name
            name = f.name
            try:
                value = getattr(self.object, f.name)
            except getattr(self.model, f.name).RelatedObjectDoesNotExist:
                value = None
            is_blank = (value is None)
        else:
            name = f.verbose_name
            value = getattr(self.object, f.name)
            is_blank = (value in f.empty_values)

        if is_blank:
            items = []
        else:
            if f.many_to_one or f.one_to_one:  # TODO: test 1-1 fields
                items = [(value, tables.get_record_url(value))]
            elif f.one_to_many or f.many_to_many:
                url_kw = {
                    'obj_model': self.object._meta.model_name,
                    'pk': self.object.pk,
                    'field': f.name,
                }
                items = [(value, reverse('relations', kwargs=url_kw))]
            elif isinstance(f, URLField):
                items = [(value, value)]
            else:
                items = self.object.get_attr_urls(f.attname)

        if hasattr(f, 'choices') and f.choices:
            display_value = getattr(self.object, f'get_{f.name}_display')()
            if is_blank:
                if display_value not in f.empty_values:
                    # let's show the display value for a blank (it could be!)
                    is_blank = False
                # assume no URL
                items = [(display_value, None)]
            else:
                if len(items) == 1:
                    items = [(display_value, items[0][1])]
                else:
                    # a bit non-sensical, we could try splitting the display
                    # value, but let's not do anything, should not ever get
                    # here
                    pass

        # only show a unit if there is a value
        unit = None if is_blank else getattr(f, 'unit', None)

        extra_info = getattr(f, 'pseudo_unit', None)
        return (name, extra_info, items, unit)

    def get_details(self):
        """
        A list, one element per field to be passed to the template

        Returns a list of 4-tuples:
          - field name
          - pseudo unit / extra field info
          - list if pairs (value item, URL)
          - real unit
        This is about the order in which things are displayed on screen
        """
        details = []
        for i in self.get_ordered_fields():
            if isinstance(i, str):
                # detail of a non-field attribute
                item = (i, None, [(getattr(self.object, i), None)], None)
                attrname = i
            else:
                item = self.get_detail_for_field(i)
                attrname = i.name

            detail_fn_name = f'get_{attrname}_detail'
            if detail_fn := getattr(self, detail_fn_name, None):
                item = detail_fn(i, item)
                if item is None:
                    continue
            details.append(item)

        if exturl := self.object.get_external_url():
            details.append(('external URL', None, [(exturl, exturl)], None))

        return details


class DatasetView(MapMixin, RecordView):
    model = models.Dataset
    template_name = 'glamr/dataset.html'
    fields = [
        'primary_ref',
        'sample',
        'bioproject',
        'gold_id',
        'jgi_project',
        'mgrast_study',
        'material_type',
        'water_bodies',
        'primers',
        'sequencing_target',
        'sequencing_platform',
        'size_fraction',
    ]

    def get_object_lookups(self):
        key = self.kwargs['key']
        if self.kwargs.get('ktype', None) == 'pk:':
            return dict(pk=key)
        else:
            # default to lookup via set number
            return dict(dataset_id=f'set_{key}')

    def get_sample_queryset(self):
        return self.object.sample_set.all()

    def get_ordered_fields(self):
        fields = []
        external_accn_fields = [
            'bioproject',
            'gold_id',
            'jgi_project',
            'mgrast_study',
        ]
        # hide these fields if they are blank
        for i in super().get_ordered_fields():
            try:
                if i.name in external_accn_fields:
                    if getattr(self.object, i.name) in i.empty_values:
                        continue
            except AttributeError:
                # wasn't a real field
                pass
            fields.append(i)
        return fields


class FrontPageView(SearchFormMixin, MapMixin, BaseMixin, SingleTableView):
    model = models.Dataset
    search_model = SearchFormMixin.ANY_MODEL
    template_name = 'glamr/frontpage.html'
    table_class = tables.DatasetTable

    filter_class = DatasetFilter
    formhelper_class = DatasetFilterFormHelper
    context_filter_name = 'filter'

    @cached_property
    def _queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(private=False)
        qs = qs.select_related('primary_ref')

        # to get sample type in table
        qs = qs.prefetch_related(Prefetch(
            'sample_set',
            queryset=Sample.objects.only('dataset_id', 'sample_type'),
        ))

        qs = qs.annotate(sample_count=Count('sample', distinct=True))

        self.filter = DatasetFilter(self.request.GET, queryset=qs)
        self.filter.form.helper = DatasetFilterFormHelper()

        return self.filter.qs.order_by("-sample_count")

    def get_queryset(self):
        # use caching as this gets called multiple times (maps etc.)
        return self._queryset

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
            ctx['search_model_name'] = self.ANY_MODEL_URL_PART
        else:
            ctx['db_is_good'] = True

        return ctx

    def _get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['show_internal_nav'] = settings.INTERNAL_DEPLOYMENT
        ctx['mc_abund'] = TaxonAbundance.objects \
            .filter(taxon__taxname__name='Microcystis') \
            .select_related('sample')[:5]

        ctx[self.context_filter_name] = self.filter

        ctx['dataset_totalcount'] = Dataset.objects.count()
        ctx['filtered_dataset_totalcount'] = self.filter.qs.distinct().count()
        ctx['sample_totalcount'] = Sample.objects.count()

        ctx['fit_map_to_points'] = False  # showing the Great Lakes

        # Compile data for dataset summary: A list of the rows of the table,
        # for each cell it's a tuple of URL query string and value (text or
        # count.)
        df = Dataset.objects.summary(
            column_field='sample_type',
            row_field='geo_loc_name',
        )
        conf = DataConfig(Dataset)
        head = []
        for i in df.columns:
            conf.filter['sample__sample_type'] = i
            head.append((conf.url_query(), i))
        dataset_counts_data = [head]
        for lake, lake_counts in df.to_dict(orient='index').items():
            conf.clear_selection()
            if lake == 'other':
                conf.q = [~ Q(sample__geo_loc_name__in=GREAT_LAKES)]
            else:
                conf.filter = dict(sample__geo_loc_name=lake)
            row = [(conf.url_query(), lake)]
            for samp_type, count in lake_counts.items():
                if count > 0:
                    conf.filter['sample__sample_type'] = samp_type
                    q_str = conf.url_query()
                else:
                    q_str = None
                row.append((q_str, count))
            dataset_counts_data.append(row)
        ctx['dataset_counts'] = dataset_counts_data

        # Compile data for sample summary: Similar to above for datasets
        df = Sample.objects.summary('sample_type', 'geo_loc_name')
        conf = DataConfig(Sample)
        head = []
        for i in df.columns:
            conf.filter['sample_type'] = i
            head.append((conf.url_query(), i))
        sample_counts_data = [head]
        for lake, lake_counts in df.to_dict(orient='index').items():
            conf.clear_selection()
            if lake == 'other':
                conf.q = [~ Q(geo_loc_name__in=GREAT_LAKES)]
            else:
                conf.filter = dict(geo_loc_name=lake)
            row = [(conf.url_query(), lake)]
            for samp_type, count in lake_counts.items():
                if count > 0:
                    conf.filter['sample_type'] = samp_type
                    q_str = conf.url_query()
                else:
                    q_str = None
                row.append((q_str, count))
            sample_counts_data.append(row)
        ctx['sample_counts'] = sample_counts_data

        return ctx


class ReferenceView(RecordView):
    """
    Display fields in roughly in bibliographic order.

    Shows related datasets at last (if any), depending if this is a primary
    paper or not.
    """
    model = models.Reference
    fields = ['authors', 'title', 'publication', 'doi', 'key_words',
              'abstract', 'dataset_primary', 'dataset']
    exclude = ['reference_id', 'short_reference']
    related = []

    def get_queryset(self):
        return super().get_queryset().prefetch_related('dataset_set')

    def get_object_lookups(self):
        key = self.kwargs['key']
        if self.kwargs.get('ktype', None) == 'pk:':
            return dict(pk=key)
        else:
            # default to lookup via paper number
            return dict(reference_id=f'paper_{key}')

    def get_publication_detail(self, field, item):
        name, info, values, unit = item
        values = [(f'{self.object.publication} ({self.object.year})', None)]
        return ('published in', info, values, unit)

    def get_dataset_primary_detail(self, field, item):
        name, info, _, _ = item
        values = [
            (i.display_simple(), get_record_url(i))
            for i in self.object.dataset_set.all()
            if i.primary_ref_id == self.object.pk
        ]
        if values:
            return (name, 'for which this is the primary paper', values, None)
        else:
            return None

    def get_dataset_detail(self, field, item):
        """
        Show any non-primary datasets
        """
        name, info, _, _ = item
        values = [
            (i.display_simple(), get_record_url(i))
            for i in self.object.dataset_set.all()
            if i.primary_ref_id != self.object.pk
        ]

        is_primary = any((
            i.primary_ref_id == self.object.pk
            for i in self.object.dataset_set.all()
        ))
        if is_primary:
            name = f'Other associated {name}'
        else:
            name = f'Associated {name}'

        if values:
            return (name, info, values, None)
        else:
            return None


class OverView(BaseMixin, SingleTableView):
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


class OverViewSamplesView(BaseMixin, SingleTableView):
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


class SampleListView(MapMixin, ModelTableMixin, BaseMixin, SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = get_sample_model()
    template_name = 'glamr/sample_list.html'
    table_class = tables.SampleTable

    def get_queryset(self):
        f = dict(dataset_id=f'set_{self.kwargs["set_no"]}')
        try:
            self.dataset = models.Dataset.objects.get(**f)
        except models.Dataset.DoesNotExist:
            raise Http404(f'no such dataset: {f=}')
        return self.dataset.samples()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['dataset'] = self.dataset
        return ctx


class SampleView(RecordView):
    model = get_sample_model()
    template_name = 'glamr/sample_detail.html'
    fields = [
        'dataset',
        'sample_name',
        'collection_timestamp',
        'geo_loc_name',
        'noaa_site',
        RecordView.OTHERS,
    ]
    exclude = Sample.get_internal_fields()
    related = []

    def get_queryset(self):
        qs = self.model.objects.with_privates()
        qs = qs.select_related('dataset', 'dataset__primary_ref')
        return qs

    def get_object_lookups(self):
        key = self.kwargs['key']
        if self.kwargs.get('ktype', None) == 'pk:':
            return dict(pk=key)
        else:
            # default to lookup via sample number
            return dict(sample_id=f'samp_{key}')

    def get_ordered_fields(self):
        fields = []
        for i in super().get_ordered_fields():
            try:
                if i.name.endswith('_loaded'):
                    continue
                if i.name.endswith('_ok'):
                    continue
                if i.one_to_many:
                    continue
            except AttributeError:
                # not a field
                pass
            fields.append(i)
        return fields

    def get_details(self):
        if self.object.private:
            # limit fields and add a special notice for private samples
            self.fields = ['sample_name', 'collection_timestamp',
                           'geo_loc_name', 'noaa_site']
            special_note = [(
                'status',
                None,
                [('sample from non-public dataset', None)],
                None
            )]
        else:
            special_note = []

        return special_note + super().get_details()

    def get_collection_timestamp_detail(self, field, item):
        name, info, _, _ = item
        value = self.object.format_collection_timestamp()
        return (name, info, [(value, None)], None)

    def get_header_links(self):
        """ a list of lists of pairs (url, link text) """
        obj = self.object
        tr = {i.flag for i in obj.tracking.all()}

        krona_url = None
        taxabund_url = None
        if not obj.private:
            if SampleTracking.Flag.TAXABUND in tr:
                krona_url = reverse(
                    'krona', kwargs=dict(samp_no=obj.get_samp_no())
                )
                taxabund_url = reverse(
                    'relations',
                    kwargs=dict(
                        obj_model='sample',
                        pk=obj.pk,
                        field='taxonabundance',
                    ),
                )
        abund_links = [
            (krona_url, 'krona chart'),
            (taxabund_url, 'abundance/taxa'),
        ]

        if obj.sample_type == 'metagenome':
            funcabund_url = None
            if not obj.private and SampleTracking.Flag.UR1ABUND in tr:
                funcabund_url = reverse(
                    'relations',
                    kwargs=dict(
                        obj_model='sample',
                        pk=obj.pk,
                        field='functional_abundance',
                    ),
                )
            abund_links.append((funcabund_url, 'abundance/functions'))

        if obj.file_set.exists():
            urlkw = dict(
                obj_model='sample',
                pk=obj.pk,
                field='file',
            )
            dl_url = reverse('relations', kwargs=urlkw)
        else:
            dl_url = None

        return [abund_links, [(dl_url, 'file downloads')]]

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['header_link_groups'] = self.get_header_links()
        return ctx


class TaxonView(RecordView):
    model = TaxNode
    fields = ['rank', 'name', 'taxid', 'parent']
    related = ['taxname', 'children']

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.prefetch_related('taxname_set', 'children__taxname_set')
        qs = qs.select_related('parent')
        return qs


class SearchView(BaseMixin, SearchFormMixin, TemplateView):
    """ display the advanced search page """
    template_name = 'glamr/search_init.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)

        # Support for uniref ID lookups
        ctx['ur90filter'] = UniRef90Filter()
        ctx['ur100filter'] = UniRef100Filter()

        # support for regular per modelfilters
        ctx['standard_filters'] = [
            (model._meta.model_name,
             model._meta.verbose_name,
             filt_cls())
            for model, filt_cls in filter_registry.primary.items()
        ]

        # Support for the Advanced Filter section
        ctx['advanced_search'] = False
        m_and_f = []
        for model, fields in search_fields.items():
            m_and_f.append((
                model._meta.model_name,
                model._meta.verbose_name,
                fields[:7],
            ))
        ctx['models_and_fields'] = m_and_f

        # models in alphabetical order
        models = get_registry().models
        items = []
        for i in ADVANCED_SEARCH_MODELS:
            m = models[i]
            items.append((m._meta.model_name, m._meta.verbose_name))
        ctx['adv_search_models'] = sorted(items, key=lambda x: x[1].lower())

        return ctx


class SearchModelView(EditFilterMixin, BaseMixin, TemplateView):
    """ offer model-based searching """
    template_name = 'glamr/search_model.html'
    model = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        return ctx


class TableView(FilterMixin, MapMixin, ModelTableMixin, BaseMixin,
                SingleTableView):
    template_name = 'glamr/filter_list.html'


class ToManyListView(FilterMixin, ModelTableMixin, BaseMixin, SingleTableView):
    """ List records related to other record """
    template_name = 'glamr/relations_list.html'

    def setup(self, request, *args, **kwargs):
        obj_model = kwargs['obj_model']
        try:
            self.obj_model = get_registry().models[obj_model]
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
        # hide the column for the object:
        self.exclude.append(self.field.remote_field.name)

        try:
            self.accessor_name = field.get_accessor_name()
        except AttributeError:
            self.accessor_name = field.name

        # add model for ModelTableMixin.setup()
        kwargs['model'] = self.model._meta.model_name
        super().setup(request, *args, **kwargs)

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(**{self.field.remote_field.name: self.object})
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object'] = self.object
        ctx['object_model_name'] = self.obj_model._meta.model_name
        ctx['object_model_name_verbose'] = self.obj_model._meta.verbose_name
        ctx['field'] = self.field
        ctx['verbose_name_plural'] = self.model._meta.verbose_name_plural
        return ctx


class SearchResultMixin(MapMixin):
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['suggestions'] = self.suggestions
        return ctx

    def get_queryset(self):
        """ returns list of 5-tuples """
        self.search_result = self.search()
        if not self.search_result:
            return []
        return self.search_result.get_object_list(self)

    def get_sample_queryset(self):
        if self.search_model == self.ANY_MODEL:
            # we won't show a map with global search, so let's skip all the
            # when after get_map_points() calls this
            return Sample.objects.none()

        sample_pks = set(self.search_result.get_pks(Sample))
        dataset_pks = self.search_result.get_pks(Dataset)
        sample_pks.update(
            Sample.objects.filter(dataset__pk__in=dataset_pks)
            .values_list('pk', flat=True)
        )
        return Sample.objects.filter(pk__in=sample_pks)


class SearchResultListView(SearchResultMixin, SearchMixin, BaseMixin,
                           ListView):
    template_name = 'glamr/result_list.html'


class AdvFilteredListView(SearchFormMixin, MapMixin, ModelTableMixin,
                          BaseMixin, SingleTableView):
    """
    View for the advanced filtering option

    Of uncertain fate, may become deprecated.
    """
    template_name = 'glamr/filter_list.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        # support searchable models only for now
        if self.model._meta.model_name not in self.model_class:
            raise Http404('model not supported in this view')
        self.conf = TableConfig(self.model)

    def set_filter(self):
        """ set filter from GET querystring """
        self.conf.clear_selection()
        self.conf.set_from_query(self.request.GET)
        try:
            # if this fails, assume it's something bad via URL
            self.conf.get_queryset()
        except Exception as e:
            raise Http404(f'suspected bad URL query string: {e}') from e
        return

    def get_queryset(self):
        # By the usual calling order set_filter() is already run by
        # get_context_data(), but just to cover the inordinary, call it here
        # again:
        self.set_filter()
        return self.conf.get_queryset()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        self.set_filter()
        ctx['filter_items'] = [
            (k.replace('__', ' -> '), v)
            for k, v in self.conf.filter.items()
        ] + [('', i) for i in self.conf.q]
        return ctx


class FilteredListView(SearchFormMixin, FilterMixin, MapMixin, ModelTableMixin,
                       BaseMixin, SingleTableView):
    """ Similar to FilteredListView but got via django-filter filters """
    template_name = 'glamr/filter_list.html'


class UniRef100View(RecordView):
    model = UniRef100

    def get_uniref90_detail(self, field, details):
        """ inject URL """
        name, info, val_items, unit = details
        val_items = [
            (val, f'https://www.uniprot.org/uniref/UniRef90_{val}')
            for val, _ in val_items
        ]
        if val_items:
            info = 'external URL'
        return (name, info, val_items, unit)


record_view_registry = DefaultDict(
    dataset=DatasetView.as_view(),
    sample=SampleView.as_view(),
    reference=ReferenceView.as_view(),
    taxnode=TaxonView.as_view(),
    uniref100=UniRef100View.as_view(),
    default=RecordView.as_view(),
)
"""
The record view registry: maps model name to RecordView subclass views.
RecordView is fall-back for models for which no key exists.  This exists purely
for record_view() so that the view function don't get computed at runtime.
"""


def record_view(*args, **kwargs):
    """
    Dispatch function to delegate to RecordView-derived view based on model
    """
    return record_view_registry[kwargs.get('model')](*args, **kwargs)


def test_server_error(request):
    if settings.ENABLE_TEST_VIEWS:
        raise RuntimeError('you were asking for it')
    else:
        raise Http404('settings.ENABLE_TEST_VIEWS is set to False')


class MiniTestView(RequiredSettingsMixin, BaseMixin, View):
    """
    minimalistic display and print request/response to stderr

    GET /minitest
    """
    required_settings = 'ENABLE_TEST_VIEWS'

    def get(self, request, *args, **kwargs):
        log.debug(f'{self}: {request=}')
        log_msg = ''
        for k, v in vars(request).items():
            if k == 'environ':
                continue  # same as META
            log_msg += f'{k}: {pprint.pformat(v)}\n'
        log.debug(log_msg)

        response = HttpResponse('ok')
        log.debug(f'{self}: {pprint.pformat(vars(response))=}')
        return response


class BaseTestView(RequiredSettingsMixin, BaseMixin, TemplateView):
    """ Display rendered base template via GET /basetest """
    required_settings = 'ENABLE_TEST_VIEWS'
    template_name = 'glamr/base.html'
