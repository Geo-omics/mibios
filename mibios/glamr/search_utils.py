"""
Stuff related to search and search suggestions
"""
from collections import namedtuple
from copy import copy
from functools import cache
from itertools import groupby
from logging import getLogger
from operator import attrgetter

from django.apps import apps
from django.conf import settings
from django.db import connections, router
from django.db.backends.signals import connection_created
from django.db.utils import OperationalError
from django.dispatch import receiver
from django.utils.safestring import mark_safe

from mibios.umrad.models import Model
from . import HORIZONTAL_ELLIPSIS
from .utils import get_record_url, verbose_field_name

log = getLogger(__name__)


spellfix_models = [
    'CompoundRecord', 'ReactionRecord', 'CompoundName',
    'FunctionName', 'Location', 'FuncRefDBEntry',
]
SPELLFIX_TABLE = 'searchterms'


@cache
def get_connection(for_write=False):
    """
    Gets the DB connection for accessing the spellfix tables

    This basically assumes that all spellfixed models must be on the same DB
    """
    models = [apps.get_model('umrad', i) for i in spellfix_models]
    if for_write:
        alias = router.db_for_write(models[0])
    else:
        alias = router.db_for_read(models[0])
    return connections[alias]


@receiver(connection_created)
def load_sqlite_spellfix_extension(
    connection=None,
    spellfix_ext_path=None,
    **kwargs,
):
    """
    Load spellfix extension for sqlite3 DBs
    """
    if connection is None:
        raise ValueError('connection parameter must not be None')

    if spellfix_ext_path:
        path = spellfix_ext_path
    else:
        path = settings.SPELLFIX_EXT_PATH

    if connection.vendor == 'sqlite' and path:
        connection.connection.load_extension(path)
        log.info('sqlite3 spellfix extension loaded')


def update_spellfix(spellfix_ext_path=None):
    """
    Populate search spellfix suggestions table

    :param str spellfix_ext_path:
        Provide path to spellfix shared object in case.  With this the spellfix
        tables can be populated before search suggestion get enabled via
        settings.

    This needs to run once, before get_suggestions() can be called.
    """
    if not settings.SPELLFIX_EXT_PATH:
        return

    Searchable = apps.get_model('glamr', 'Searchable')
    connection = get_connection(for_write=True)

    if spellfix_ext_path is not None:
        connection.ensure_connection()
        load_sqlite_spellfix_extension(
            connection=connection,
            spellfix_ext_path=spellfix_ext_path,
        )

    with connection.cursor() as cur:
        cur.execute('BEGIN')
        cur.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS '
                    f'{SPELLFIX_TABLE} USING spellfix1')
        cur.execute(f'DELETE FROM {SPELLFIX_TABLE}_vocab')
        log.info('spellfix table deleted')
        cur.execute(
            'INSERT INTO {spellfix_table}(word) SELECT {field} '
            'FROM {table} WHERE {field} NOT IN '
            '(SELECT word FROM {spellfix_table}_vocab)'.format(
                spellfix_table=SPELLFIX_TABLE,
                field='text',
                table=Searchable._meta.db_table,
            )
        )
        cur.execute('COMMIT')
        log.info('spellfix table populated')


def get_suggestions(query):
    """
    Get spelling suggestions

    Returns a dict mapping each word of the query to a list of suggestions.  An
    empty such list indicates the word is spelled correctly.
    """
    match get_connection().vendor:
        case 'sqlite':
            # single word spell-checking only
            return {query: get_suggestions_sqlite(query)}
        case 'postgresql':
            UniqueWord = apps.get_model('glamr', 'UniqueWord')
            return UniqueWord.objects.all().suggest(query)
        case _:
            return []


def get_suggestions_sqlite(query):
    if not settings.SPELLFIX_EXT_PATH:
        return []

    with get_connection().cursor() as cur:
        try:
            cur.execute(
                f'select word from {SPELLFIX_TABLE} where word match %s '
                f'and scope=4',  # small speed-up
                [query]
            )
        except OperationalError as e:
            if e.args[0] == f'no such table: {SPELLFIX_TABLE}':
                log.warning('Search suggestions were not set up, '
                            'update_spellfix() needs to be run!')
            else:
                raise
        return [i[0] for i in cur.fetchall()]


class SearchHitObj:
    """
    Full-text search hit for a single object

    Helper class for full-text search results.
    """
    Subhit = namedtuple('Subhit', ['field', 'snippet', 'url'], defaults=[None])

    def __init__(self, pk, content_type_id, model):
        self.is_first = False  # is 1st item for its content type/model
        self.model = model  # the model class
        self.pk = pk  # the object's PK
        self.obj = None
        self.subhits = []  # a list of field, snippet pairs
        self.rank = -1  # the maximum rank of any of the field/snippet pairs
        self.content_type_id = content_type_id  # pk in contenttype table

    def add(self, hit):
        """
        Add a hit

        hit: a Searchable with rank/snippet annotations
        """
        if hit.object_id != self.pk:
            raise ValueError('pk mismatch')
        if hit.content_type_id != self.content_type_id:
            raise ValueError('content_type_id mismatch')

        if hasattr(hit, 'snippet'):
            # ts_highlight() individually marks consequtive words
            # of a matching phrase, let's highlight the whole
            # phrase
            # TODO: improve this for when snippet has symbols, e.g.
            # "LE18-22.4"
            snippet = hit.snippet.replace('</mark> <mark>', ' ')
            # add ... if snippet is inside the text
            plain = snippet.replace('<mark>', '').replace('</mark>', '')  # noqa:E501
            if not hit.text.startswith(plain):
                snippet = f'{HORIZONTAL_ELLIPSIS} {snippet}'
            if not hit.text.endswith(plain):
                snippet = f'{snippet} {HORIZONTAL_ELLIPSIS}'
        else:
            # highlighting was OFF
            snippet = hit.text

        # assuming text was html escaped before stored to the DB
        snippet = mark_safe(snippet)

        self.subhits.append(self.Subhit(hit.field, snippet))
        self.rank = max(self.rank, hit.rank)

    def copy(self):
        """ make a deep copy for get_object_list() """
        obj = copy(self)
        obj.subhits = obj.subhits[:]
        return obj


class SearchResult(dict):
    """
    Result of a full-text search

    Helper class for full-text search results.
    """
    @classmethod
    def from_searchables(cls, qs, model_cache):
        """
        Return a new instance build from a SearchableQueryset

        qs - assumed to be a RawQueryset ordered by content_type_id
        """
        data = {}

        top_rank = {}  # remembers max rank for model
        for ctype_id, grp in groupby(qs, key=attrgetter('content_type_id')):
            model = model_cache[ctype_id]
            data[model] = {}
            top_rank[model] = -1
            for i in grp:
                if i.object_id not in data[model]:
                    data[model][i.object_id] = \
                            SearchHitObj(i.object_id, i.content_type_id, model)
                data[model][i.object_id].add(i)
                top_rank[model] = \
                    max(top_rank[model], data[model][i.object_id].rank)

        # forget pk keys and order mode-wise by top rank
        _data = (
            (model, list(data[model].values()))
            for model, _
            in sorted(top_rank.items(), key=lambda x: -x[1])
        )
        return cls(_data)

    def get_pks(self, model):
        """
        Get object PKs for a model's hits

        Returns an empty list if hits for this model were found
        """
        return [i.pk for i in self.get(model, [])]

    def get_object_list(self, view=None):
        """
        Return object_list suitable for a SearchResultMixin(ListView)

        view: instance of SearchResultMixin, may be None for testing.
        """
        self.view = view
        ret = []
        for model, hits in self.items():
            ret += list(self.object_list_items(model))
        return ret

    DETAIL_FIELDS = {
        'dataset': ('primary_ref', 'scheme', 'water_bodies',
                    'material_type',),
        'sample': ('dataset', 'geo_loc_name', 'sample_type',
                   'collection_timestamp', 'project_id'),
    }
    """ extra details for some models to show when displaying results """

    _object_cache = {}
    """ class-level object cache """

    @classmethod
    def update_object_cache(cls, model, pks):
        """ get objects corresponding to our hits """
        if model not in cls._object_cache:
            cls._object_cache[model] = {}
        cache = cls._object_cache[model]
        missing_pks = [i for i in pks if i not in cache]
        qs = model.objects.all()
        if model._meta.model_name == 'taxnode':
            # do not incur extra query to display each node's name
            # (per TaxNode.__str__)
            # TODO: other models with potentially 1000s of search hits may need
            # something like this too?
            qs = qs.prefetch_related('taxname_set')
        elif model._meta.model_name == 'dataset':
            qs = qs.select_related('primary_ref')
        elif model._meta.model_name == 'sample':
            qs = qs.select_related('dataset__primary_ref')
        cache.update(qs.in_bulk(missing_pks))

    def object_list_items(self, model):
        """
        Turn SearchMixin.search_result into something suitable for
        ListView.object_list.

        Helper to get_objects_list() returning the items for a single model.

        This may query the DB to match hit with corresponding objects.
        """
        self.update_object_cache(model, (i.pk for i in self[model]))
        objs = self._object_cache[model]

        if self.view is None or self.view.search_model is self.view.ANY_MODEL:
            detail_fields = {}
        else:
            detail_fields = self.DETAIL_FIELDS.get(model._meta.model_name, {})

        is_first = True  # True for the first item of each model
        for obj_hit in self[model]:
            obj_hit = obj_hit.copy()

            if obj_hit.pk not in objs:
                # search index out-of-sync, just ignore (what else?)
                continue

            obj_hit.is_first = is_first
            obj_hit.obj = objs[obj_hit.pk]
            is_first = False

            # For certain models add more details to the subhit list.
            # First get field values, with URLs for FKs, skip blanks,
            # then overwrite with snippets:
            details = {}
            for attr in detail_fields:
                value = getattr(obj_hit.obj, attr)
                details[attr] = SearchHitObj.Subhit(
                    attr,
                    value,
                    get_record_url(value) if isinstance(value, Model) else None,  # noqa:E501
                )
            for subhit in obj_hit.subhits:
                attr = subhit.field
                if attr in details:
                    # use the snippet
                    details[attr] = details[attr]._replace(snippet=subhit.snippet)  # noqa:E501
                else:
                    # append snippet
                    details[attr] = subhit

            # replace field attr name with verbose field name
            # (breaks if this method is called a second time)
            details = [
                i._replace(field=verbose_field_name(
                    model._meta.model_name, i.field
                ))
                for i in details.values()
            ]
            obj_hit.subhits = details

            if not settings.INTERNAL_DEPLOYMENT:
                # don't display rank on production site
                obj_hit.rank = None

            yield obj_hit

    def get_stats(self):
        return [(model, len(hits)) for model, hits in self.items()]
