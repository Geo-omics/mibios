from functools import cache
import re

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
            return obj.get_record_url(obj, ktype=None)
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
