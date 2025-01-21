from collections import OrderedDict
import csv
import io
from itertools import islice, tee, zip_longest
from math import isnan
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from django.apps import apps
from django.db import OperationalError
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from django.http import Http404, HttpResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils.html import format_html
from django.views.decorators.cache import cache_control, cache_page
from django.views.generic.base import ContextMixin, TemplateView, View
from django.views.generic.edit import FormView

from django_tables2 import (SingleTableMixin, SingleTableView, Column,
                            MultiTableMixin)

from pandas import isna
import zipstream

from mibios.ops.utils import profile_this
from . import (__version__, QUERY_FORMAT, QUERY_AVG_BY,
               get_registry)
from .data import DataConfig, TableConfig, NO_CURATION_PREFIX
from .forms import (ExportForm, get_field_search_form, UploadFileForm,
                    ShowHideForm)
from .load import Loader
from .management.import_base import AbstractImportCommand
from .models import ChangeRecord, ImportFile, Snapshot
from .model_graphs import get_model_graph_info
from .tables import (DeletedHistoryTable, HistoryTable,
                     CompactHistoryTable, DetailedHistoryTable,
                     SnapshotListTable, SnapshotTableColumn, Table,
                     table_factory, ORDER_BY_FIELD)
from .utils import get_db_connection_info, getLogger


log = getLogger(__name__)


class ExportCSVDialect(csv.unix_dialect):
    """ export data csv dialect """
    delimiter = '\t'
    quoting = csv.QUOTE_NONE


class SearchFieldLookupError(LookupError):
    pass


class CuratorMixin():
    CURATOR_GROUP_NAME = 'curators'
    user_is_curator = False

    def setup(self, request, *args, **kwargs):
        f = dict(name=self.CURATOR_GROUP_NAME)
        try:
            self.user_is_curator = request.user.groups.filter(**f).exists()
        except Exception:
            pass  # default applies
        super().setup(request, *args, **kwargs)


class UserRequiredMixin(LoginRequiredMixin):
    raise_exception = True
    permission_denied_message = 'You don\'t have an active user account here.'


class CuratorRequiredMixin(CuratorMixin, UserRequiredMixin,
                           UserPassesTestMixin):
    permission_denied_message = 'You are not a curator'

    def test_func(self):
        return self.user_is_curator


class StaffLoginRequiredMixin(UserPassesTestMixin):
    raise_exception = True
    permission_denied_message = 'Login via staff account is required.'

    def dispatch(self, request, *args, **kwargs):
        disp = super().dispatch
        if request.user.is_authenticated:
            disp = cache_control(private=True)(disp)
        return disp(request, *args, **kwargs)

    def test_func(self):
        return self.request.user.is_staff


class VersionInfoMixin:
    """
    inject some debug info into template context

    Use together with a ContextMixin
    """
    def get_version_info(self):
        """
        Compile some debugging info that can be inserted as comment into an
        html response.
        """
        info = {
            'mibios': __version__,
        }
        for conf in get_registry().apps.values():
            info[conf.name] = getattr(conf, 'version', None)
        if settings.DEBUG:
            info['DEBUG'] = 'True'
        for db_alias, db_info in get_db_connection_info().items():
            info[f'DB {db_alias}'] = db_info
        info['view'] = self.__class__.__name__
        info['template'] = getattr(self, 'template_name', '???')
        return info

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['version_info'] = self.get_version_info()
        return ctx


class BasicBaseMixin(CuratorMixin, VersionInfoMixin, ContextMixin):
    """
    Mixin to populate context for the base template without model/dataset info
    """
    def setup(self, request, *args, **kwargs):
        log.debug(request.resolver_match or 'no url resolver match')
        log.debug(f'user:{request.user} path:{request.path_info} '
                  f'GET:{request.GET}')
        request.GET = self.parse_query_string_csv(request.GET)
        super().setup(request, *args, **kwargs)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # page_title: a list, inheriting views should consider adding to this
        ctx['page_title'] = [getattr(
            get_registry(),
            'verbose_name',
            apps.get_app_config('mibios').verbose_name
        )]
        ctx['user_is_curator'] = self.user_is_curator
        return ctx

    @staticmethod
    def parse_query_string_csv(get):
        """
        convert CSVs to lists in query string

        Such CSVs come back via saving lists with MultipleHiddenInput

        :param get QueryDict: Usually the GET attribute of a request.

        Returns a modified QueryDict instance.  This should be called as the
        first thing in setup().
        """
        # FIXME: allow values to contain properly escaped commas, currently
        # everything with a comma gets turned into a list
        get = get.copy()
        for k, vs in get.lists():
            get.setlist(
                k,
                # flatten list of CSVs:
                [val for i in vs for val in i.split(',')]
            )
        get._mutable = False
        return get

    def dispatch(self, request, *args, **kwargs):
        if 'nocache' in request.GET:
            log.debug('SKIPPING CACHE')
            return super().dispatch(request, *args, **kwargs)
        else:
            return cache_page(None)(super().dispatch)(request, *args, **kwargs)


class BaseMixin(BasicBaseMixin):
    """
    Mixin to populate context for the base template
    """
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        reg = get_registry()
        ctx['model_names'] = OrderedDict()
        ctx['data_names'] = OrderedDict()

        for app_name, app_conf in reversed(list(reg.apps.items())):
            if app_conf.name == 'mibios':
                verbose_name = 'auxiliary'
            else:
                verbose_name = app_conf.verbose_name

            model_names = sorted((
                (i._meta.model_name, i._meta.verbose_name_plural)
                for i in reg.get_models(app=app_conf.name)
            ))
            if model_names:
                ctx['model_names'][verbose_name] = model_names

            data_names = sorted(
                get_registry().get_dataset_names(app=app_conf.name)
            )
            if data_names:
                ctx['data_names'][app_conf.verbose_name] = data_names

        try:
            ctx['snapshots_exist'] = Snapshot.objects.exists()
        except OperationalError as e:
            log.error(f'BaseMixin.get_context_data: {type(e)}: {e}')
            ctx['snapshots_exist'] = False

        ctx['site_name'] = reg.verbose_name
        return ctx


class DatasetMixin(BaseMixin):
    """
    Mixin for views that deal with one dataset/model

    The url to which the inheriting view responds must supply a 'data_name'
    kwarg that identifies the dataset or model.
    """
    config_class = DataConfig
    show_hidden = False
    data_name = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # variables set by setup():
        self.conf = None

    def setup(self, request, *args, data_name=None, **kwargs):
        """
        Set up dataset/model attributes of instance

        This overrides (but calls first) View.setup()
        """
        super().setup(request, *args, **kwargs)

        is_curated = True
        if data_name is not None:
            if data_name.startswith(NO_CURATION_PREFIX):
                if not hasattr(self, 'user_is_curator'):
                    raise Http404

                if self.user_is_curator:
                    is_curated = False
                else:
                    raise PermissionDenied

                data_name = data_name[len(NO_CURATION_PREFIX):]
            self.data_name = data_name

        if self.data_name is None:
            # FIXME: may this happen?
            return

        try:
            self.conf = self.config_class(
                self.data_name,
                show_hidden=self.show_hidden,
            )
        except LookupError:
            # no table or data set with that name
            raise Http404(f'no dataset with name: {self.data_name}')

        self.conf.is_curated = is_curated

        # this sort of assumes that all requests are GET
        self.process_query_string()
        log.debug(f'CONF = {vars(self.conf)}')

    def process_query_string(self):
        """
        Load the query string / request.GET into data config

        Method should be called ahead of get().  When overriding this method to
        allow special processing, the child class should first call the
        parent's method.
        """
        if self.conf:
            self.conf.set_from_query(self.request.GET)

    @classmethod
    def shorten_lookup(cls, txt):
        """
        Abbreviate a lookup string for display

        Turns "foo__bar__field_name" into "f-n-field_name".  Simple field names
        are left as-is.
        """
        *rels, field = txt.split('__')
        if rels:
            rels = '-'.join([j[0] for j in rels]) + '-'
        else:
            rels = ''
        return rels + field

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.conf is None:
            return ctx

        ctx['model'] = self.conf.model._meta.model_name
        if self.conf.is_curated:
            ctx['url_data_name'] = self.conf.name
        else:
            ctx['url_data_name'] = NO_CURATION_PREFIX + self.conf.name
        ctx['data_name'] = self.conf.name
        ctx['page_title'].append(self.conf.verbose_name)
        ctx['data_name_verbose'] = self.conf.verbose_name

        ctx['applied_filter'] = [
            (k, v, self.conf.remove_filter(**{k: v}))
            for k, v
            in self.conf.filter.items()
        ]
        ctx['applied_excludes_list'] = [
            (i, self.conf.remove_excludes(i))
            for i
            in self.conf.excludes
        ]

        # the original querystring to be appended to various URLs:
        querystr = self.conf.url_query()
        ctx['querystr'] = '?' + querystr if querystr else ''

        ctx['avg_by_data'] = {
            '-'.join(i): [self.shorten_lookup(j) for j in i]
            for i in self.conf.model.average_by
        }

        return ctx


class TableViewPlugin():
    """
    Parent class for per-model view plugins

    The table view plugin mechanism allows to add per-model content to the
    regular table display.  An implementation should declare a class in a
    registered app's view module that inherits from TableViewPlugin and that
    should set the model_class and template_name variables and override
    get_context_data() as needed.  The template with the to-be-added content
    needs to be provided at the app's usual template location.  The plugin's
    content will be rendered just above the table.
    """
    model_class = None
    template_name = None

    def __init__(self, view):
        self.view = view
        self.setup()

    def setup(self):
        pass

    def get_queryset(self, qs):
        return qs

    def get_context_data(self, **ctx):
        return ctx


class TableView(DatasetMixin, UserRequiredMixin, SingleTableView):
    template_name = 'mibios/table.html'
    config_class = TableConfig

    # Tunables adjusting display varying on number of unique values:
    MEDIUM_UNIQUE_LIMIT = 10
    HIGH_UNIQUE_LIMIT = 30

    def __init__(self, *args, **kwargs):
        self.compute_counts = False
        self.plugin = None
        super().__init__(*args, **kwargs)

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            plugin_class = \
                get_registry().table_view_plugins[self.conf.name]
        except KeyError:
            self.plugin = None
        else:
            self.plugin = plugin_class(view=self)

    def get_queryset(self):
        if hasattr(self, 'object_list'):
            return self.object_list

        if self.conf is None:
            return []

        qs = self.conf.get_queryset()
        if self.plugin:
            qs = self.plugin.get_queryset(qs)
        return qs

    def get_table_class(self):
        t = table_factory(conf=self.conf)
        return t

    def get_sort_by_field(self):
        """
        Returns name of valid sort-by fields from the querystring

        If the sort-by field is not a field in the current table view None is
        returned.
        """
        field = self.request.GET.get(ORDER_BY_FIELD)
        if not field:
            return None

        field = field.lstrip('-')
        # reverse from django_tables2 accessor sep
        field = field.replace('.', '__')
        if field in self.conf.fields:
            return field

        return None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.conf is None:
            return ctx

        ctx['field_search_form'] = None
        sort_by_field = self.get_sort_by_field()
        if sort_by_field is None:
            ctx['sort_by_stats'] = {}
        else:
            # prepare all things needed for the selected/sorted column:
            add_search_form = True
            ctx['sort_by_field'] = sort_by_field
            qs = self.get_queryset()
            stats = qs.get_field_stats(sort_by_field, natural=True)
            if 'uniform' in stats or 'unique' in stats:
                try:
                    del stats['choice_counts']
                    del stats['description']
                except KeyError:
                    pass

                if 'uniform' in stats:
                    add_search_form = True

            else:
                # a non-boring column
                if 'description' in stats:
                    # only give these for numeric columns
                    try:
                        if stats['description'].dtype.kind == 'f':
                            # keep description and only give NaNs as filter
                            # choice
                            try:
                                nan_ct = stats['choice_counts'][[float('nan')]]
                            except KeyError:
                                del stats['choice_counts']
                            else:
                                stats['choice_counts'] = nan_ct
                        else:
                            del stats['description']
                    except KeyError:
                        pass

                filter_link_data = []
                if 'choice_counts' in stats:
                    if len(stats['choice_counts']) > self.MEDIUM_UNIQUE_LIMIT:
                        add_search_form = True
                    if len(stats['choice_counts']) < self.HIGH_UNIQUE_LIMIT:
                        # link display gets unwieldy at high numbers
                        counts = {
                            None if isinstance(k, float) and isnan(k) else k: v
                            for k, v in
                            stats['choice_counts'].items()
                        }
                        filter_link_data = [
                            (
                                # '' => None hack for blank char fields to make
                                # dash appear
                                None if isna(value) or value == '' else value,
                                count,
                                # TODO: applying filter to negated queryset is
                                # more complicated
                                self.conf.add_filter(**{sort_by_field: value}),
                                self.conf.add_exclude(**{sort_by_field: value}),  # noqa: E501
                            )
                            for value, count
                            in counts.items()
                        ]
                ctx['filter_link_data'] = filter_link_data
            ctx['sort_by_stats'] = stats

            if sort_by_field is not None:
                try:
                    if self.conf.model.is_bool_field(sort_by_field):
                        # search fields on boolean don't make sense
                        add_search_form = False
                except LookupError:
                    # fields that are not real fields
                    pass

            if add_search_form:
                try:
                    ctx['field_search_form'] = \
                        get_field_search_form(
                            self.conf,
                            *self.get_search_field()
                    )()
                except SearchFieldLookupError:
                    pass
            # END sort-column stuff

        ctx['curation_switch_conf'] = \
            self.conf.put(is_curated=not self.conf.is_curated)

        ctx['count_switch_conf'] = \
            self.conf.put(with_counts=not self.conf.with_counts)

        # Plugin: process the plugin last so that their get_context_data() gets
        # a full view of the existing context:
        if self.plugin:
            ctx['table_view_plugin_template'] = self.plugin.template_name
            ctx = self.plugin.get_context_data(**ctx)
        else:
            ctx['table_view_plugin_template'] = None

        # add relation links
        related_confs = []
        for i in self.conf.model.get_fields(with_reverse=True).fields:
            if not self.conf.model.is_relation_field(i):
                continue
            try:
                related_confs.append(self.conf.shift(i))
            except NotImplementedError:
                # FIXME: remove this after shift is fully implemented
                pass
        ctx['related_confs'] = related_confs

        return ctx

    def get_search_field(self):
        """
        Helper to figure out on which field to search

        Returns a non-empty list of field names, else raises
        SearchFieldLookupError to indicate that the current sort-by column has
        no corresponding field(s) on which to perform a search, i.e.  no search
        form should be displayed.
        """
        name = self.get_sort_by_field()
        try:
            field = self.conf.model.get_field(name)
        except LookupError as e:
            # name column is assumed to be natural key
            if name == 'name':
                field = None
                model = self.conf.model
            else:
                raise SearchFieldLookupError from e
        else:
            if field.name == 'id':
                return [field.name]

            if field.is_relation:
                model = field.related_model
            else:
                return [field.name]

        try:
            kw = model.natural_lookup(None)
        except Exception as e:
            raise SearchFieldLookupError from e

        return [(field.name + '__' if field else '') + i for i in kw.keys()]


class Values2CSVGenerator:
    """
    Helper to turn Table.as_values() data into http streaming response bytes

    Performance:  The outer iterator collects up to chunk_size lines and yields
    them in one chunk.  Yielding about 1000 lines at a time gives us about 8
    MB/s up from 2 MB/s if yielding single lines.

    We attempt to detect and log client disconnection.  There is
    instrumentation that logs total transfer stats and data rate.
    """
    default_chunk_size = 1000

    def __init__(self, rows, sep, chunk_size=default_chunk_size):
        """
        Parameters to set up the csv generator:

        values:
            an iterable over the rows, e.g. Table.as_values(), each row must be
            an iterable over the indivual values, which may be of any (simple)
            type.
        sep: The column separator, e.g. '\t' or ','
        """
        self.chunk_size = chunk_size
        self.rows = rows
        self.sep = sep
        self.num_rows = 0
        self.num_chunks = 0
        self.total_bytes = 0
        self.total_time = None

    def close(self):
        """
        wsgi might call this in certain error cases

        We'll try to log what error is being handled by wsgi.
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            log.error(
                f'Export aborted! cause: {exc_type.__name__}: {exc_value}\n'
                f'A traceback follows.  If additionally a full traceback is '
                f'printed to stdout/err, then this indicates a bug in mibios/'
                f'glamr.  Absense of the extra stack trace indicates that WSGI'
                f' aborted the export, possibly for client disconnection.'
            )
            log.info(''.join(traceback.format_tb(exc_traceback)))

        self.rows.close()

    def __iter__(self):
        t0 = time.monotonic()
        try:
            with profile_this(enabled=False, name='format_rows'):
                yield from self._format_rows()
                # yield from self._format_rows_csv()
        finally:
            self.total_time = time.monotonic() - t0
            rate = self.total_bytes / self.total_time / 1_000_000
            megabytes = self.total_bytes / 1_000_000
            log.info(f'export rows:{self.num_rows}({self.num_chunks})'
                     f'/{megabytes:.1f}M @ {rate:.1f}M/s')

    def _format_rows(self):
        """
        Row csv formatter

        This variant with hand-crafted formatting is less versatile but a bit
        faster (~5%) than the csv module based alternative.
        """
        sjoin = self.sep.join
        while True:
            data = b''.join((
                f'{sjoin(("" if i is None else str(i) for i in row))}\n'.encode()  # noqa:E501
                for row in islice(self.rows, self.chunk_size)
            ))
            chunk_bytes = len(data)
            if not chunk_bytes:
                # the final chunk is always zero bytes
                break
            self.num_chunks += 1
            self.num_rows += self.chunk_size  # likely wrong at last chunk
            self.total_bytes += chunk_bytes
            yield data

    def _format_rows_csv(self):
        """
        Alternative implementation using the stdlib's csv module
        """
        buf = io.StringIO()
        writer = csv.writer(buf, dialect=ExportCSVDialect, delimiter=self.sep)
        while True:
            writer.writerows(islice(self.rows, self.chunk_size))
            chunk_bytes = buf.truncate()
            if not chunk_bytes:
                # the final chunk is always zero bytes
                break
            self.num_chunks += 1
            self.num_rows += self.chunk_size  # likely wrong at last chunk
            self.total_bytes += chunk_bytes
            buf.seek(0)
            yield buf.getvalue().encode()
        buf.close()


class BaseRenderer:
    """
    Abstract helper class to generate an http response to data export / file
    download requests

    Sub-classes must implement a render() method that takes an iterator over
    chunks of data, e.g. a table row.  For a regular http response render()
    must write the content to the response object.  For streaming, a generator
    function is passed to the response object.
    """
    description = None
    content_type = None
    streaming_support = None

    def __init__(self, filename):
        if self.streaming_support:
            response_class = StreamingHttpResponse
        else:
            response_class = HttpResponse

        self.response = response_class(content_type=self.content_type)
        self.response['Content-Disposition'] = \
            f'attachment; filename="{filename}"'

    def render(self, values):
        raise NotImplementedError


class CSVRenderer(BaseRenderer):
    description = 'comma-separated text file'
    content_type = 'text/csv'
    delimiter = ','
    streaming_support = True

    def render(self, values):
        if self.response.streaming:
            return self._render_as_stream(values)
        else:
            return self._render_no_stream(values)

    def _render_nostream(self, values):
        """
        Render all rows to the response
        """
        writer = csv.writer(self.response, delimiter=self.delimiter,
                            lineterminator='\n')
        for i in values:
            writer.writerow(i)

    def _render_as_stream(self, values):
        """
        Make a response content generator, rendering all rows.

        values: An iterable over rows, each row being a list of str values.

        The content generator yields bytes.
        """
        self.stream_generator = Values2CSVGenerator(values, self.delimiter)
        self.response.streaming_content = self.stream_generator
        return

        def _render():
            CHUNK_SIZE = 1000
            rows = (f'{self.delimiter.join(row)}\n'.encode() for row in values)
            while True:
                data = b''.join(islice(rows, CHUNK_SIZE))
                if not data:
                    break
                yield data

        self.response.streaming_content = _render()


class CSVTabRenderer(CSVRenderer):
    description = '<tab>-separated text file'
    delimiter = '\t'


class CSVRendererZipped(CSVRenderer):
    description = ('comma-separated text file', 'zipped')
    content_type = 'application/zip'

    def __init__(self, response, filename):
        super().__init__(filename=filename[:-len('.zip')])

    def _render(self, values):
        """
        render the csv
        """
        for row in values:
            yield b'\t'.join([str(i).encode() for i in row])
            yield b'\n'

    def render(self, values):
        """
        Set response's streaming content to rendered data
        """
        z = zipstream.ZipFile(compression=ZIP_DEFLATED)
        z.write_iter(self.filename, self._render(values))
        self.response.streaming_content = z


class CSVTabRendererZipped(CSVRendererZipped):
    description = ('<tab>-separated text file', 'zipped')
    delimiter = '\t'


class TextRendererZipped(BaseRenderer):
    description = ('text file', 'zipped')
    content_type = 'application/zip'
    streaming_support = False

    def __init__(self, filename):
        super().__init__(filename=filename[:-len('.zip')])

    def render(self, values):
        """
        Render all rows to the response

        :param values: An iterator over str lines ending with a newline.
        """
        with ZipFile(self.response, 'w', ZIP_DEFLATED) as z:
            with z.open(self.filename, 'w') as f:
                for line in values:
                    f.write(line.encode())


class ExportBaseMixin:
    # Supported export format registry
    # (name, file suffix, renderer class)
    FORMATS = (
        ('csv', '.csv', CSVRenderer),
        ('tab', '.csv', CSVTabRenderer),
        ('csv/zipped', '.csv.zip', CSVRendererZipped),
        ('tab/zipped', '.csv.zip', CSVTabRendererZipped),
    )
    DEFAULT_FORMAT = 'csv'

    def get_format(self, fmt_name=None):
        """
        Get the requested export file format
        """
        if fmt_name is None:
            fmt_name = self.request.GET.get(QUERY_FORMAT)
        for i in self.FORMATS:
            if fmt_name == i[0]:
                return i

        if fmt_name:
            raise Http404('unknown export format')

        for i in self.FORMATS:
            if self.DEFAULT_FORMAT == i[0]:
                return i
        else:
            raise RuntimeError('no valid default export format defined')

    def get_format_choices(self):
        """ Return split choices and defaults for formating options """
        f_opts = set()
        c_opts = set()
        for opt, _, render_cls in self.FORMATS:
            f, _, c = opt.partition('/')
            descr = render_cls.description
            if isinstance(descr, tuple):
                f_descr = render_cls.description[0]
                c_descr = render_cls.description[1]
            elif isinstance(descr, str):
                f_descr = descr
                if c:
                    raise ValueError('description of deflate option missing')
                else:
                    c_descr = 'uncompressed'

            f_opts.add((f, f_descr))
            c_opts.add((c, c_descr))

        f_default, _, c_default = self.DEFAULT_FORMAT.partition('/')
        if f_default not in [i[0] for i in f_opts]:
            raise ValueError('format default is not a valid choice')
        if c_default not in [i[0] for i in c_opts]:
            raise ValueError('deflate default is not a valid choice')

        return {
            'choices': {
                'format': sorted(f_opts),
                'deflate': sorted(c_opts),
            },
            'defaults': {
                'format': f_default,
                'deflate': c_default,
            },
        }


class ExportMixin(ExportBaseMixin):
    """
    Export table data as file download

    A mixin for TableView

    Requires kwargs['format'] to be set by url conf.

    Implementing views need to provide a get_values() method that provides the
    data to be exported as an iterable over rows (which are lists of values).
    The first row should contain the column headers.
    """
    def get_filename(self):
        """
        Get the user-visible name (stem) for the file downloaded.

        The default implementation generates a default value from the registry
        name.  The returnd value is without suffix.  The suffix is determined
        by the file format.
        """
        return get_registry().name + '_data'

    def render_to_response(self, context):
        name, suffix, Renderer = self.get_format()
        filename = self.get_filename() + suffix

        renderer = Renderer(filename=filename)
        renderer.render(self.get_values())
        return renderer.response


class ExportView(ExportMixin, TableView):
    """
    File download of table data
    """
    def get_filename(self):
        return self.conf.name

    def get_values(self):
        return self.get_table().as_values()


class ExportFormView(ExportBaseMixin, DatasetMixin, FormView):
    """
    Provide the export format selection form

    The form will be submitted via GET and the query string language used, is
    what TableView expects.  So this will not use Django's usual form
    processing.
    """
    template_name = 'mibios/export.html'
    export_url_name = 'export'
    config_class = TableConfig  # need show attribute

    def get_form_class(self):
        return ExportForm.factory(self)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['page_title'].append('export')
        ctx['avg_url_slug'] = None
        return ctx


class ImportView(DatasetMixin, CuratorRequiredMixin, FormView):
    template_name = 'mibios/import.html'
    form_class = UploadFileForm
    log = getLogger('dataimport')
    show_hidden = True

    def form_valid(self, form):
        # do data import
        f = form.files['file']
        note = form.cleaned_data['note']
        dry_run = form.cleaned_data['dry_run']
        if dry_run:
            log.debug(
                '[dry run] Importing into {}: {}'.format(self.conf.name, f)
            )
        else:
            self.log.info(
                'Importing into {}: {}'.format(self.conf.name, f)
            )

        try:
            stats = Loader.load_file(
                f,
                self.conf.name,
                dry_run=dry_run,
                can_overwrite=form.cleaned_data['overwrite'],
                erase_on_blank=form.cleaned_data['erase_on_blank'],
                warn_on_error=True,
                no_new_records=not form.cleaned_data['allow_new_records'],
                user=self.request.user,
                note=note,
            )

        except Exception as e:
            if settings.DEBUG:
                raise
            msg = ('Failed to import data in uploaded file: {}: {}'
                   ''.format(type(e).__name__, e))
            msg_level = messages.ERROR
        else:
            import_log = ''
            if note:
                import_log += f' Note: {note}\n\n'
            import_log += AbstractImportCommand.format_import_stats(
                **stats,
                verbose_changes=True,
            )

            msg = 'data successfully imported'
            msg_level = messages.SUCCESS

            msg_log = msg + '\n{}' + import_log

            file_rec = stats.get('file_record', None)
            if file_rec is None:
                msg_html = msg + ', log:<br><pre>{}</pre>'
                msg_html = format_html(msg_html, import_log)
            else:
                file_rec.log = import_log
                file_rec.save()
                msg_html = msg + ', see details at <a href="{}">import log</a>'
                url = reverse('log', kwargs=dict(import_file_pk=file_rec.pk))
                msg_html = format_html(msg_html, url)
        finally:
            f.close()

        messages.add_message(self.request, msg_level, msg_html)
        args = (msg_level, 'user:', self.request.user, 'file:', f, '\n',
                msg_log)
        if dry_run:
            log.log(*args)
        else:
            self.log.log(*args)

        return super().form_valid(form)

    def get_success_url(self):
        # TODO: return self.conf.url()
        return reverse('table', kwargs=dict(data_name=self.conf.name))

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['page_title'].append('file upload')
        # col_names are for django_tables2's benefit, so we need to use the
        # field names if the col name is None:
        ctx['col_names'] = [
            (j if j else i.capitalize())
            for i, j in zip(self.conf.fields, self.conf.fields_verbose)
        ]
        try:
            dataset = get_registry().datasets[self.conf.name]
        except KeyError:
            ctx['dataset_doc'] = None
        else:
            ctx['dataset_doc'] = dataset.get_doc()
        return ctx


class ShowHideFormView(DatasetMixin, FormView):
    template_name = 'mibios/show_hide_form.html'

    def get_form_class(self):
        self.initial = {'show': tuple(self.conf.fields)}
        return ShowHideForm.factory(self.conf)


class HistoryView(BaseMixin, CuratorRequiredMixin, MultiTableMixin,
                  TemplateView):
    """
    Show table with history of a single record
    """
    table_class = HistoryTable
    record = None
    template_name = 'mibios/history.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        if 'record' in kwargs:
            # coming in through mibios.ModelAdmin history
            self.record = kwargs['record']
            self.record_pk = self.record.pk
            self.record_natural = self.record.natural
            self.model_name = self.record._meta.model_name
        else:
            # via other url conf, NOTE: has no current users
            try:
                self.record_pk = int(kwargs['natural'])
                self.record_natural = None
            except ValueError:
                self.record_pk = None
                self.record_natural = kwargs['natural']

            self.model_name = kwargs['data_name']
            try:
                model_class = get_registry().models[self.model_name]
            except KeyError:
                raise Http404

        if self.record is None:
            get_kw = {}
            if self.record_natural:
                get_kw['natural'] = self.record_natural
            elif self.record_pk:
                get_kw['pk'] = self.record_pk

            try:
                self.record = model_class.objects.get(**get_kw)
            except (model_class.DoesNotExist,
                    model_class.MultipleObjectsReturned):
                self.record = None

        if kwargs.get('extra_context'):
            if self.extra_context is None:
                self.extra_context = kwargs['extra_context']
            else:
                self.extra_context.update(kwargs['extra_context'])

    def get_tables(self):
        """
        Get the regular history and a table of lost/missing
        """
        tables = []
        regular = self.record.history.select_related('user', 'file')
        tables.append(self.table_class(self._add_diffs(regular)))

        # get lost or otherwise extra
        reg_pks = (i.pk for i in regular)
        f = dict(
            record_type__model=self.model_name,
        )
        if self.record_natural:
            f['record_natural'] = self.record_natural
        elif self.record_pk:
            f['record_pk'] = self.record_pk

        extra = ChangeRecord.objects.exclude(pk__in=reg_pks).filter(**f)
        if extra.exists():
            tables.append(self.table_class(self._add_diffs(extra)))
        return tables

    def _add_diffs(self, qs):
        """
        Add differences to history queryset

        :param qs: iterable of ChangeRecord instances.  All changes must belong
                   to the same record and be in order.
        """
        # diffs for each and precessor, compare itertools pairwise recipe:
        a, b = tee(qs)
        next(b, None)  # shift forward, diff to last/None will give all fields
        diffs = []
        for i, j in zip_longest(a, b):
            d = i.diff_to(j)
            diffs.append(d)

        # combine into data
        data = list(qs)
        for diff, row in zip(diffs, data):
            row.changes = diff
        return data

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.record:
            natural_key = self.record.natural
        else:
            # TODO: review this use-case and maybe get the key from table data
            natural_key = '???'
        ctx['natural_key'] = natural_key
        ctx['page_title'].append(f'history of {natural_key}')

        return ctx


class DeletedHistoryView(BaseMixin, CuratorRequiredMixin, SingleTableView):
    template_name = 'mibios/deleted_history.html'
    table_class = DeletedHistoryTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        try:
            model = get_registry().models[kwargs['data_name']]
        except KeyError:
            raise Http404

        try:
            # record_type: can't name this content_type, that's taken in
            # TemplateResponseMixin
            self.record_type = ContentType.objects.get_by_natural_key(
                model._meta.app_label,
                model._meta.model_name,
            )
        except ContentType.DoesNotExist:
            raise Http404

    def get_queryset(self):
        if not hasattr(self, 'object_list'):
            f = dict(
                is_deleted=True,
                record_type=self.record_type,
            )
            self.object_list = ChangeRecord.objects.filter(**f)

        return self.object_list

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['record_model'] = self.record_type.name
        ctx['page_title'].append('deleted records')
        return ctx


class CompactHistoryView(BaseMixin, UserRequiredMixin, SingleTableMixin,
                         TemplateView):
    table_class = CompactHistoryTable
    template_name = 'mibios/compact_history.html'

    def get_table_data(self):
        return ChangeRecord.summary_dict()


class DetailedHistoryView(BaseMixin, UserRequiredMixin, SingleTableMixin,
                          TemplateView):
    table_class = DetailedHistoryTable
    template_name = 'mibios/detailed_history.html'

    def dispatch(self, request, *args, **kwargs):
        self.first = kwargs['first']
        self.last = kwargs['last']
        return super().dispatch(request, *args, **kwargs)

    def get_table_data(self):
        return ChangeRecord.get_details(self.first, self.last)


class FrontPageView(BaseMixin, UserRequiredMixin, SingleTableMixin,
                    TemplateView):
    template_name = 'mibios/frontpage.html'
    table_class = CompactHistoryTable

    def get_table_data(self):
        return ChangeRecord.summary_dict(limit=5)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['counts'] = {}
        models = get_registry().get_models()

        def sort_key(m):
            return m._meta.verbose_name_plural.casefold()

        for i in sorted(models, key=sort_key):
            count = i.objects.count()
            ctx['counts'][i._meta.verbose_name_plural.capitalize()] = count

        ctx['have_changes'] = ChangeRecord.objects.exists()
        ctx['admins'] = settings.ADMINS
        return ctx


class SnapshotListView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    View presenting a list of snapshots with links to SnapshotView
    """
    model = Snapshot
    table_class = SnapshotListTable


class SnapshotView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    View of a single snapshot, displays the list of available tables
    """
    template_name = 'mibios/snapshot.html'

    def get_table_class(self):
        meta_opts = dict(
            # model=self.model,
            # template_name='django_tables2/bootstrap.html',
        )
        Meta = type('Meta', (object,), meta_opts)
        table_opts = dict(Meta=Meta)
        table_opts.update(table=SnapshotTableColumn(self.snapshot.name))
        name = ''.join(self.snapshot.name.split()).capitalize()
        name += 'SnapshotTable'
        # FIXME: call django_tables2.table_factory??
        klass = type(name, (Table,), table_opts)
        return klass

    def get(self, request, *args, **kwargs):
        try:
            self.snapshot = Snapshot.objects.get(name=kwargs['name'])
        except Snapshot.DoesNotExist:
            raise Http404

        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        return self.snapshot.get_table_name_data()


class SnapshotTableView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    Display one table from a snapshot (with all data)
    """
    template_name = 'mibios/snapshot_table.html'

    def get(self, request, *args, **kwargs):
        snapshot = kwargs['name']
        self.app_label = kwargs['app']
        self.table_name = kwargs['table']
        try:
            self.snapshot = Snapshot.objects.get(name=snapshot)
        except Snapshot.DoesNotExist:
            raise Http404

        try:
            self.columns, rows = \
                self.snapshot.get_table_data(self.app_label, self.table_name)
        except ValueError:
            # invalid table name
            raise Http404

        self.queryset = [dict(zip(self.columns, i)) for i in rows]

        return super().get(request, *args, **kwargs)

    def get_table_class(self):
        meta_opts = dict()
        Meta = type('Meta', (object,), meta_opts)
        table_opts = dict(Meta=Meta)
        for i in self.columns:
            table_opts.update(**{i: Column()})
        name = ''.join(self.snapshot.name.split()).capitalize()
        name += 'SnapshotTableTable'
        # FIXME: call django_tables2.table_factory??
        klass = type(name, (Table,), table_opts)
        return klass


class ExportSnapshotTableView(ExportMixin, SnapshotTableView):
    def get_filename(self):
        return self.snapshot.name + '_' + self.table_name

    def get_values(self):
        return self.get_table().as_values()


class ImportFileDownloadView(CuratorRequiredMixin, View):
    """
    Reply to file download request with X-Sendfile headed response
    """
    def get(self, request, *args, **kwargs):
        path = 'imported/' + str(kwargs['year']) + '/' + kwargs['name']
        try:
            file = ImportFile.objects.get(file=path)
        except ImportFile.DoesNotExist:
            raise Http404
        res = HttpResponse(content_type='')
        res['X-Sendfile'] = str(file)
        return res


class AverageMixin():
    """
    Add to TableView to display tables with averages
    """
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        # Collect the avg-by fields from both, the path and the querystring,
        # since both are in use by different views.  At some point we'll
        # probably drop putting these into the url path just not yet.
        avg_by = []
        if 'avg_by' in kwargs:
            avg_by = kwargs['avg_by'].split('-')

        if QUERY_AVG_BY in self.conf.extras:
            for i in self.conf.extras[QUERY_AVG_BY]:
                if i not in avg_by:
                    avg_by.append(i)

            del self.conf.extras[QUERY_AVG_BY]

        for i in self.conf.model.average_by:
            # TODO: is this check really useful? should it be done here?
            if set(avg_by).issubset(set(i)):
                break
        else:
            raise Http404(f'unexpected avg_by: {self.conf.avg_by}')

        # Use fields that average() adds:
        fields = avg_by + ['avg_group_count']
        fields += [i.name for i in self.conf.model.get_average_fields()]

        self.conf.avg_by = avg_by
        self.conf.fields = fields
        self.conf.fields_verbose = [None] * len(self.conf.fields)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        avg_by = self.conf.avg_by
        ctx['avg_url_slug'] = '-'.join(avg_by)
        ctx['avg_by_short'] = [self.shorten_lookup(i) for i in avg_by]
        ctx['page_title'].append('average')
        return ctx


class AverageView(AverageMixin, TableView):
    pass


class AverageExportView(AverageMixin, ExportView):
    pass


class AverageExportFormView(AverageMixin, ExportFormView):
    export_url_name = 'average_export'


class LogView(BaseMixin, CuratorRequiredMixin, TemplateView):
    template_name = 'mibios/log.html'

    def get(self, request, *args, import_file_pk=None, **kwargs):
        try:
            self.import_file = ImportFile.objects.get(pk=import_file_pk)
        except ImportFile.DoesNotExist:
            raise Http404
        return super().get(request, *args, **kwargs)


class ModelGraphView(TemplateView):
    template_name = 'mibios/model_graphs.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.available_apps = [
            i for i in get_registry().apps.keys()
            if i != 'mibios'
        ]
        if 'app_name' in kwargs:
            self.app_name = kwargs['app_name']
            try:
                self.image_file_name = get_model_graph_info()[self.app_name]
            except KeyError:
                raise Http404('invalid app name')

        else:
            # None triggers displaying the app list
            self.app_name = None
            self.image_file_name = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['app_name'] = self.app_name
        ctx['image_file_name'] = self.image_file_name
        ctx['available_apps'] = list(get_model_graph_info().keys())
        return ctx
