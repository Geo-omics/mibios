"""
Stuff related to search and search suggestions
"""
from functools import cache
from logging import getLogger

from django.conf import settings
from django.db import connections, router
from django.db.backends.signals import connection_created
from django.db.utils import OperationalError
from django.dispatch import receiver
from django.utils.html import escape as html_escape
from django.utils.safestring import mark_safe

from mibios.omics import get_sample_model
from mibios.umrad.models import (
    CompoundRecord, CompoundName, FunctionName, Location,
    FuncRefDBEntry, ReactionRecord, Uniprot, UniRef100,
)
from .models import Searchable, UniqueWord

log = getLogger(__name__)


spellfix_models = [
    CompoundRecord, ReactionRecord, CompoundName,
    FunctionName, Location, FuncRefDBEntry,
]
searchable_models = spellfix_models + [
    UniRef100, Uniprot, get_sample_model(),
]
SPELLFIX_TABLE = 'searchterms'


@cache
def get_connection(for_write=False):
    """
    Gets the DB connection for accessing the spellfix tables

    This basically assumes that all spellfixed models must be on the same DB
    """
    if for_write:
        alias = router.db_for_write(spellfix_models[0])
    else:
        alias = router.db_for_read(spellfix_models[0])
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


def highlight(query, document):
    if connections['default'].vendor == 'postgresql':
        return highlight_postgresql(query, document)
    else:
        # not implemented; return document unchanged
        return document


def highlight_postgresql(query, document, search_type='websearch'):
    FUNCS = dict(
        websearch='websearch_to_tsquery',
    )
    SQL = 'SELECT ts_headline(%(document)s, {to_tsquery_func}(%(query)s))'

    try:
        to_tsquery_func = FUNCS[search_type]
    except KeyError:
        raise ValueError('search type not supported: {search_type}')

    sql = SQL.format(to_tsquery_func=to_tsquery_func)
    params = dict(
        query=query,
        document=html_escape(document),
    )

    with connections['default'].cursor() as cur:
        cur.execute(sql, params)
        res = list(cur.fetchall())
        assert len(res) == 1, f'A: got more (or less) than expected: {res=}'
        assert len(res[0]) == 1, f'B: got more (or less) than expected: {res=}'
        # return a str
        return mark_safe(res[0][0])
