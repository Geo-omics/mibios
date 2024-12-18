"""
Utilities module
"""
from datetime import datetime
import inspect
import logging
import time

from django.conf import settings
import django.db


logging.addLevelName(25, 'SUCCESS')


class PrintLikeLogging(logging.LoggerAdapter):
    """
    Adapter to log just like print
    """
    def log(self, level, *msg, sep=' ', **kwargs):
        """
        Adapter to log just like print

        Except end, file, and flush keyword args are not to be used
        """
        super().log(level, sep.join([str(i) for i in msg]), **kwargs)


def getLogger(name):
    """
    Wrapper around logging.getLogger()
    """
    return PrintLikeLogging(logging.getLogger(name), {})


class QueryLogFilter(logging.Filter):
    """
    Filter to add call origin info to the db.backends debug log entries.

    With this we can tell where in our code DB queries originate.  It doesn't
    work for queries that are triggered by the django machinery after our code
    has already done it's work, e.g. the usual call to get_queryset() in a
    class-based view that is triggered by the response rendering.  Some of our
    middleware might still show up in those queries' call traces but those get
    skipped here as to not clutter the logs.

    The filter should be added to the django.db.backends logger during the app
    initilization in DEBUG mode.
    """
    # declare which modules are interesting call sites and what to skip
    GOOD_MODULE_PREFIXES = ['mibios', 'django_tables2']
    SKIP_THESE = {
        ('mibios.models', 'count'): 1,
        # ('django_tables2.data', '__len__'): 2,
        # ('django_tables2.rows', '__len__'): 2,
    }

    def filter(self, record):
        call_site_msg = ''
        skipped = ''
        fr = inspect.currentframe()
        while True:
            # iterate up the call chain, ends when f_back is None
            if fr.f_back is None:
                break

            fr = fr.f_back
            mod = inspect.getmodule(fr)
            if mod is None:
                # maybe we're interactive or so?
                continue

            mname = mod.__name__

            for i in self.GOOD_MODULE_PREFIXES:
                if mname.startswith(i):
                    break
            else:
                # e.g. we're deep in the django framework
                continue

            info = inspect.getframeinfo(fr)

            if skip_id := self.SKIP_THESE.get((mname, info.function), None):
                skipped += f'+{skip_id}'
                continue

            if mname in ['mibios.ops.utils', 'mibios.utils']:
                if info.function == '__call__':
                    # skip middleware frames
                    continue

            call_site_msg = f' via: {mname}:{info.lineno} {info.function}()'
            break

        if skipped:
            call_site_msg += ' ' + skipped

        record.msg += call_site_msg
        return True


class DeepRecord():
    """
    Dict-of-dict / tree like datastructure to hold a multi-table record

    Helps loading complex table
    """
    SEP = '__'

    def __init__(self, init={}, sep=SEP):
        self._ = {}
        self.sep = sep
        for k, v in init.items():
            self.add(k, v)

    @classmethod
    def from_accessors(cls, accessors, sep='__'):
        """
        Make a deep template dict for the model(s)
        """
        return cls(init={i: {} for i in accessors}, sep=sep)

    def split(self, key):
        """
        ensure key is in split format and valid for other methods
        """
        if isinstance(key, str):
            key = key.split(self.sep)
        return key

    def __getitem__(self, key):
        """
        Get method for dict for dicts / a.k.a. deep model template

        key can be a __-separated string (django lookup style) or a list
        of dict keys
        """
        cur = self._
        key = self.split(key)

        for i in key:
            try:
                cur = cur[i]
            except (KeyError, TypeError):
                raise LookupError('Invalid key: {}'.format(key))
        return cur

    def __delitem__(self, key):
        """
        Del method for dict for dicts / a.k.a. deep model template

        key can be a __-separated string (django lookup style) or a list
        of dict keys
        """
        # FIXME: has currently no users
        cur = self._
        key = self.split(key)

        prev = cur
        for i in key:
            prev = cur
            try:
                cur = cur[i]
            except (KeyError, TypeError):
                raise KeyError('Invalid key for template: {}'.format(key))

        del prev[i]

    def add(self, key, value={}):
        """
        Add a new key with optional value

        Adds a new key with optional value. Key can be a __-separated string
        (django lookup style) or a list of dict keys
        """
        self.__setitem__(key, value)

    def __contains__(self, key):
        """
        Contains method for dict for dicts / a.k.a. deep model template

        Key can be a __-separated string (django lookup
        style) or a list
        of dict keys
        """
        cur = self._
        key = self.split(key)

        for i in key:
            try:
                cur = cur[i]
            except (KeyError, TypeError):
                return False
        return True

    def __setitem__(self, key, value):
        """
        Set method for dict for dicts / a.k.a. deep model template

        The key must exist.  Key can be a __-separated string (django lookup
        style) or a list
        of dict keys
        """
        cur = self._
        prev = None
        key = self.split(key)

        for i in key:
            prev = cur
            try:
                cur = cur[i]
            except KeyError:
                # extend key space
                cur[i] = {}
                cur = cur[i]
            except TypeError:
                # value at cur already set
                raise KeyError('Invalid key: {}, a value has already been set:'
                               ' {}'.format(key, cur))

        prev[i] = value

    def keys(self, key=(), leaves_first=False, leaves_only=False):
        """
        Return (sorted) list of keys
        """
        ret = []
        cur = self[key]
        if isinstance(cur, dict):
            if key:
                if not (cur and leaves_only):
                    ret.append(key)
            for k, v in cur.items():
                ret += self.keys(
                    key + (k,),
                    leaves_first=leaves_first,
                    leaves_only=leaves_only,
                )
        else:
            if key:
                ret.append(key)

        if leaves_first:
            ret = sorted(ret, key=lambda x: -len(x))

        return ret

    def items(self, **kwargs):
        return [(i, self[i]) for i in self.keys(**kwargs)]

    def update(self, *args, **kwargs):
        for i in args:
            for k, v in i.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __iter__(self):
        return iter(self.keys(leaves_first=True))

    def pretty(self, indent=(0, 2)):
        """
        Pretty-print object
        """
        if len(indent) == 2:
            offset, indent = indent
        elif len(indent) == 1:
            offset = 0
        else:
            raise ValueError('bad value for indent parameter: {}'
                             ''.format(indent))
        lines = []
        for k, v in self.items():
            line = '{}{}{}:'.format(
                ' ' * offset,
                ' ' * indent * (len(k) - 1),
                k[-1],
            )
            if isinstance(v, dict):
                if not v:
                    # empty leaf
                    line += ' -'
            else:
                line += '[{}] "{}"'.format(type(v).__name__, v)
            lines.append(line)
        return '\n'.join(lines)

    def __str__(self):
        return str(self._)

    def flatten(self):
        return {
            self.SEP.join(k): v
            for k, v in self.items(leaves_only=True)
        }


class StatsMiddleWare:
    def __init__(self, get_response):
        self.get_response = get_response
        self.count = 0
        self.log = getLogger('mibios.stats')

    def __call__(self, request):
        self.count += 1
        t0 = datetime.now()
        pt0 = time.process_time()
        response = self.get_response(request)
        clock = (datetime.now() - t0).total_seconds()
        proc = time.process_time() - pt0
        msg = f'request: {self.count} clock: {clock:.3f} proc: {proc:.3f}'
        self.log.info(msg)
        return response


def prep_url_query_value(value):
    """
    Mangle python objects for serialization into URL query string values

    This helper is to be used on values before calling QueryDict.setlist or
    similar.  Numbers and strings don't require special treatment.  This
    function implements transforming tuples and lists into comma-separated
    lists.  Other python objects are p[assed through without change.
    """
    if isinstance(value, (tuple, list)):
        value = ','.join((str(i) for i in value))
    return value


def url_query_value_to_python(key, value):
    """
    Convert a url query value into a python object

    This is the reverse of prep_url_query_value.  Depending on the lookup, some
    lists and tuples are converted.  All other values are passed through
    unchanged.  In particular, numeric values, can be left as strings, because
    lookup in query set filter methods are usually converted correctly
    depending of the field type in question.
    """
    if key.endswith('__in'):
        # a list
        value = value.split(',')
    elif key.endswith('__range'):
        # a tuple
        value = tuple(value.split(','))

    return value


def get_db_connection_info():
    """
    compile a information on DB connections

    Returns dict mapping the db alias to a string.  This information should not
    be displayed in production, only use this during development.
    """
    info = {}
    for i in django.db.connections.all():
        params = dict(i.get_connection_params())
        try:
            del params['password']
        except KeyError:
            pass
        if settings.DEBUG:
            params = [
                f'{k}={v}' for k, v in params.items()
                if k != 'password'
            ]
            params = ' '.join(params)
        else:
            # in production only display DB name
            params = f'database={params.get("database", "???")}'
        info[i.alias] = f'{i.display_name} ({params})'
    return info


def sort_models_by_fk_relations(models):
    """
    Get list of models sorted by FK dependency

    Enforces a partial order on the given models.  Models with foreign key
    fields go after any FK's related model.  The returned list may contain
    additional models that are discovered by following foreign key
    relations and which were not in the original list.

    It is assumed that no cycles via FK relations are encountered and nothing
    is done to detect such cycles.  Expect some infinite recursion in this
    case.
    """
    models = list(models)
    in_order = []
    while models:
        m = models.pop()
        fwd_new = set((
            i.related_model for i in m._meta.get_fields()
            if i.many_to_one
            # TODO: one_to_one?  (but need a test case)
            # but skip the following:
            and not m == i.related_model  # FK on self
            and i.related_model is not None  # GenericFK
            and i.related_model not in in_order  # already seen
        ))
        if fwd_new:
            fwd_new_sorted = sort_models_by_fk_relations(fwd_new)
        else:
            fwd_new_sorted = []
        in_order = in_order + fwd_new_sorted + [m]
        for i in fwd_new_sorted:
            if i in models:
                models.remove(i)
    return in_order
