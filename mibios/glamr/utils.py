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
    # If object/model has its own get_record_url the use that, otherwise fall
    # back to the generic 'record' url.
    if len(args) == 1 and not kwargs:
        # called with object
        obj = args[0]
        if hasattr(obj, 'get_record_url'):
            return obj.get_record_url(obj)
        model_name = obj._meta.model_name
        key = obj.pk
        ktype = None
    elif len(args) == 1 and len(kwargs) == 1:
        # called with model name and keytype/key keyword arg
        model_name = args[0]
        (ktype, key), *_ = kwargs.items()
    elif len(args) == 2 and not kwargs:
        # called with model name and key / default keytype
        model_name, key = args
        ktype = None
    else:
        raise TypeError(
            f'bad number/combination of args/kwargs: {args=} {kwargs=}'
        )

    model = get_registry()[model_name]
    if hasattr(model, 'get_record_url'):
        return model.get_record_url(key, ktype=ktype)
    else:
        kwargs = dict(model=model_name, key=key)
        if ktype is not None:
            kwargs['ktype'] = ktype + ':'
        return reverse('record', kwargs=kwargs)


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


class FKCache(dict):
    MISSING = object()

    def __init__(self, model, fk_fields=None):
        """
        Values cache for tables

        Parameters:
        model: model of table
        fk_fields:
            ForeignKey fields belonging to the model for which values shall be
            cached.
        """
        if fk_fields is None:
            fk_fields = [i for i in model._meta.get_fields() if i.many_to_one]

        for i in fk_fields:
            self[i] = {}

        self.field_names = [i.name for i in fk_fields]
        self.fk_id_attrs = [i.attname for i in fk_fields]
        self.querysets = []
        for i in fk_fields:
            manager = i.related_model.objects
            if hasattr(manager, 'str_only'):
                self.querysets.append(manager.str_only())
            else:
                self.querysets.append(manager.all())

    def update_chunk(self, queryset):
        missing_all = [set() for _ in self]
        per_rel_things = list(zip(
            self.field_names,
            self.fk_id_attrs,
            missing_all,
            self.querysets,
            self.values(),  # the per-relation caches
        ))
        # 1. get missing FKs
        for obj in queryset:
            for _, attname, missing, _, fcache in per_rel_things:
                fk = getattr(obj, attname)
                if fk is None:
                    continue
                if fk not in fcache:
                    missing.add(fk)

        # 2. get missing and update cache
        for _, _, missing, qs, fcache in per_rel_things:
            for obj in qs.filter(pk__in=missing):
                fcache[obj.pk] = obj
        del missing_all

        # 3. populate chunk with related objects
        for obj in queryset:
            for fname, attname, _, _, fcache in per_rel_things:
                if (pk := getattr(obj, attname)) is not None:
                    setattr(obj, fname, fcache[pk])
