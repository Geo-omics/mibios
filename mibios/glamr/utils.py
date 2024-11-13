from functools import cache
import re

from django.apps import apps
from django.db import connection
from django.urls import reverse

from mibios import get_registry


query_word_pat = re.compile(r"""("[^"]*"|'[^']*'|\S+)\s*""")


def split_query(query, keep_quotes=False):
    """
    Split a search query into word-like things, respecting quotes
    """
    words = []
    while query:
        match = query_word_pat.match(query)
        if not match:
            break
        word = match.group(1)  # match w/o trailing space
        query = query[match.end():]

        if not keep_quotes:
            match word[0]:
                case '"':
                    word = word.strip('"')
                case "'":
                    word = word.strip("'")
        if word:
            words.append(word)
    return words


def get_record_url(*args, **kwargs):
    """
    Return detail URL for any object

    Arguments: <obj> | <model_name, key> | <model_name, key_type=key>

    The object can be passed as the only argument.  Or the model/model name and
    PK/accession must be passed.  Only a single accession value can be passed.

    Use this instead of Model.get_absolute_url() because it needs to work on
    models from other apps.
    """
    DEFAULT_KEYTYPE = 'natkey'
    ktype = DEFAULT_KEYTYPE
    # If object/model has its own get_record_url then use that, otherwise fall
    # back to the generic 'record' url.
    if len(args) == 1 and not kwargs:
        # called with object
        obj = args[0]
        if hasattr(obj, 'get_record_url'):
            return obj.get_record_url(obj)
        model_name = obj._meta.model_name
        key = obj.pk
    elif len(args) == 1 and len(kwargs) == 1:
        # called with model name and keytype/key keyword arg
        model_name = args[0]
        (ktype, key), *_ = kwargs.items()
        ktype = ktype or DEFAULT_KEYTYPE
    elif len(args) == 2 and not kwargs:
        # called with model name and key / default keytype
        model_name, key = args
    else:
        raise TypeError(
            f'bad number/combination of args/kwargs: {args=} {kwargs=}'
        )

    model = get_registry()[model_name]
    if hasattr(model, 'get_record_url'):
        return model.get_record_url(key, ktype=ktype)
    else:
        if ktype != DEFAULT_KEYTYPE:
            key = f'{ktype}:{key}'
        return reverse('record', args=[model_name, key])


@cache
def verbose_field_name(model, accessor):
    if isinstance(model, str):
        model = get_registry()[model]
    return model.get_field(accessor).verbose_name


def estimate_row_totals(model):
    """
    get an estimate of the total number of rows in a table
    """
    if connection.vendor == 'postgresql':
        stat_model_name = 'pg_class'
    elif connection.vendor == 'sqlite':
        stat_model_name = 'dbstat'
    else:
        raise RuntimeError('db vendor not supported')

    stat_model = apps.get_model('glamr', stat_model_name)
    return int(stat_model.objects.get(name=model._meta.db_table).num_rows)


def exclude_private_data(queryset, user):
    """ helper to remove private data from generic querysets """
    qs = queryset
    if hasattr(qs, 'exclude_private'):
        # e.g. Dataset or Sample models
        return qs.exclude_private(user)

    Sample = apps.get_model('glamr', 'Sample')
    Dataset = apps.get_model('glamr', 'Dataset')

    for field in qs.model._meta.get_fields():
        if not field.many_to_one:
            continue

        if field.related_model in (Sample, Dataset):
            break
    else:
        # has no FK to Sample or Dataset
        return qs

    return qs.filter(**{
        f'{field.name}__pk__in':
        field.related_model.objects.get_allowed_pks(user)}
    )
