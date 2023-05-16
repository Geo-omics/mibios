from django import template
from django.db.models import Field, Q
from django.template.defaultfilters import capfirst, stringfilter


register = template.Library()


@register.filter(name='append')
def append(value, arg):
    """
    append argument to a comma-separated list

    The existing value may be None, in which case a new list is started and arg
    is the first and only element
    """
    item = str(arg)
    if ',' in item:
        raise ValueError('list separator (comma) in argument')
    if value is None:
        return str(item)
    else:
        return f'{value},{item}'


human_lookups = dict((
    ('icontains', 'contains (case-insensitive)'),
    ('contains', 'contains'),
    ('iexact', '= (case-insensitive)'),
    ('exact', '='),
    ('in', 'in'),
    ('gt', '>'),
    ('gte', '>='),
    ('lt', '<'),
    ('lte', '<='),
    ('istartswith', 'startswith (case-insensitive)'),
    ('startswith', 'startswith'),
    ('iendswith', 'endswith (case-insensitive)'),
    ('endswith', 'endswith'),
    ('range', 'range'),
    ('year', 'year'),
    ('month', 'month'),
    ('iregex', 'regex (case-insensitive)'),
    ('regex', 'regex'),
    ('isnull', '<blank>'),
))


@register.filter(name='qformat')
def qformat(value):
    """ format a Q filter tuple """
    key, rhs = value
    key = key.split('__')
    if key[-1] in Field.get_lookups():
        lookup = key[-1]
        lhs = key[:-1]
    else:
        lookup = 'exact'
        lhs = key

    lookup = human_lookups.get(lookup, lookup)
    return f'{"->".join(lhs)} ({lookup}) {rhs}'


@register.filter(name='is_q')
def is_q(value):
    """ check if object is a Q instance """
    return isinstance(value, Q)


@register.filter(name='capfirstsmart')
@stringfilter
def capfirstsmart(value):
    """
    Like capfirst but preserves existing capitalizations in first word.

    "pH average" -> "pH average"
    "average pH" -> "Average pH"
    """
    if value == '':
        return value
    return capfirst(value) if value.split(maxsplit=1)[0].islower() else value


@register.filter(name='meta')
def meta_attr_access(model, attrname):
    """
    Access an attribute of Model._meta
    """
    return getattr(model._meta, attrname)
