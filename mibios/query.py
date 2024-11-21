from collections import OrderedDict
from decimal import Decimal
from inspect import getgeneratorstate, getgeneratorlocals
import json
from logging import getLogger
from operator import attrgetter, itemgetter
from pathlib import Path
from pprint import pprint, pformat

from django.core.exceptions import FieldDoesNotExist
from django.db import models

from django_tables2 import table_factory
import pandas
from pandas.api.types import is_numeric_dtype


log = getLogger(__name__)


class NaturalValuesIterable(models.query.ValuesIterable):
    """
    Iterable like that returned by QuerySet.values() yielding natural values

    Replaces primary keys with their natural values
    """
    pk_fields = []
    model_class = None

    def __iter__(self):
        value_maps = {}

        # for each "field/model" generate a mapping from the "real" values to
        # the natural key for each object/row:
        for field in self.pk_fields:
            m = {}
            model = self.model_class.get_field(field).related_model
            if self.queryset.is_curated():
                manager = model.curated
            else:
                manager = model.objects
            for i in manager.iterator():
                m[i.pk] = i.natural
            value_maps[field] = m

        # Apply the mapping to each row:
        for row in super().__iter__():
            for field, m in value_maps.items():
                val = row[field]
                if val is None:
                    continue
                row[field] = m[row[field]]
            yield row


def natural_values_iterable_factory(model_class, *pk_fields):
    """
    Get custom iterable class to be used for QuerySet.values() results

    Instantiation of the iterable is a django internal so this function
    customizes the iterable class to our QuerySet.  The class should be
    assigned to QuerySet._iterator_class during values() calls.
    """
    return type(
        'CustomNaturalValuesIterable',
        (NaturalValuesIterable, ),
        dict(model_class=model_class, pk_fields=pk_fields),
    )


class Q(models.Q):
    """
    A thin wrapper around Q to handle natural lookups

    Ideally, "natural" should be a custom Lookup and then handled closer to
    the lookup-to-SQL machinery but it's not clear how to build a lookup that
    can be model-dependent expressed purely in terms of other lookups and
    doesn't touch SQL in at all.  Hence, we re-write natural lookups before
    they get packaged into the Q object.
    """
    NOT = 'NOT'

    def __init__(self, *args, model=None, **kwargs):
        # handling natural lookups is only done if the model is provided,
        # since we need to know to which model the Q is relative to
        if model is not None:
            kwargs = model.resolve_natural_lookups(**kwargs)
        super().__init__(*args, **kwargs)

    def serialize(self, as_dict=False, separators=(',', ':'), **kwargs):
        """
        Serialize a Q object to json

        This is a wrapper around json.dumps.

        :param bool as_dict: return intermediate dict representation
        """
        node = {}
        if self.negated:
            node['_ne'] = True

        if self.connector == Q.OR:
            node['_or'] = True
        elif self.connector != Q.AND:
            raise ValueError('connector must be either AND or OR')

        child_nodes = []
        for i in self.children:
            if isinstance(i, Q):
                child_nodes.append(i.serialize(as_dict=True))
            elif isinstance(i, tuple) and len(i) == 2:
                # key:val pair
                node[i[0]] = i[1]
            else:
                raise ValueError('child node must be Q object or key:value')

        if child_nodes:
            node['children'] = child_nodes

        if as_dict:
            return node
        else:
            return json.dumps(node, separators=separators, **kwargs)

    @classmethod
    def deserialize(cls, text=None, dict_repr=None):
        if text is None:
            if dict_repr is None:
                raise ValueError('need some non-None input')
            node = dict_repr
        else:
            if dict_repr is not None:
                raise ValueError('exactly one of text or dict_repr must be '
                                 'None')
            node = json.loads(text)

        conn = Q.AND
        negated = False
        child_nodes = []

        if '_or' in node:
            if node.pop('_or'):
                conn = Q.OR
            else:
                raise ValueError('_or value expected to be True')

        if '_ne' in node:
            if node.pop('_ne'):
                negated = True
            else:
                raise ValueError('_ne value expected to be True')

        if 'children' in node:
            child_nodes = [
                cls.deserialize(dict_repr=i) for i in node.pop('children')
            ]

        return cls(
            *child_nodes,
            _connector=conn,
            _negated=negated,
            **node,  # remaining leaf nodes
        )

    def resolve_path(self, path):
        """
        Get a list of Q object descendends following the given path

        The path parameter is a list of integer child indicies. For a valid
        path of length n a list of n+1 nodes is returned.  The last node
        returned is a 2-tuple, the others are Q objects.

        A path with invalid indices will raise an IndexError.  A path longer
        than the depth of the tree will raise an AttributeError.
        """
        if path:
            return [self] + self._resolve_path(path)
        else:
            return [self]

    def _resolve_path(self, path):
        """ recursive helper method for path resolution """
        first, *rest = path
        try:
            node = self.children[first]
        except IndexError as e:
            raise LookupError('invalid path') from e
        if rest:
            if not isinstance(node, Q):
                raise LookupError('invalid path, leaf was reached midway')
            nodes = node._resolve_path(rest)
        else:
            nodes = []
        return [node] + nodes

    def add_condition(self, lhs, rhs, path=[]):
        """ Insert an lhs=rhs filter condition into the tree at given node """
        obj = self & Q()
        *head, node = obj.resolve_path(path)

        if not isinstance(node, Q):
            # a leaf / (lhs, rhs)-tuple, make it into a Q object first
            parent = head[-1]  # parent must exist as old node is a leaf
            # use connector dual to the parent's, if this was not appropriate,
            # then the condition should have been added to the parent.
            if parent.connector == Q.AND:
                conn = Q.OR
            elif parent.connector == Q.OR:
                conn = Q.AND
            else:
                raise ValueError('invalid connector')
            node = Q(node, _connector=conn)
            # replace old node with new node
            index = path[-1]
            parent.children[index] = node

        node.children.append((lhs, rhs))
        return obj

    def replace_node(self, other, path):
        """ replace node at end of path with other """
        if not path:
            if isinstance(other, Q):
                return other
            else:
                # assume (lhs, rhs) tuple
                return Q(other)

        obj = self & Q()
        *_, parent, _ = obj.resolve_path(path)
        index = path[-1]
        parent.children[index] = other
        return obj

    def remove_node(self, path):
        """
        remove a node

        The path argument is the list of children indices that lead to the
        node to be removed.
        """
        if not path:
            # remove it all, also resets connector + negation state to default
            return Q()

        obj = self & Q()
        *path, index = path
        *_, node = obj.resolve_path(path)
        node.children.pop(index)
        return obj

    def negate_node(self, path):
        """
        negate the node given by path

        The path argument is the list of children indices that lead to the
        node to be negated.
        """
        obj = self & Q()  # get deep copy to return
        if not path:
            obj.negate()
            return obj

        *_, parent, node = obj.resolve_path(path)
        idx = path[-1]

        if isinstance(node, Q):
            node.negate()
            # squash if possible
            if node.connector == parent.connector and not node.negated:
                parent.children[idx:idx + 1] = node.children
            elif not node.negated and len(node.children) == 1:
                # single child, connector does not matter
                parent.children[idx] = node.children[0]
        else:
            # assume a leaf, i.e. a 2-tuple
            key, value = node
            parent.children[idx] = ~Q(**{key: value})

        return obj

    def flip_node(self, path=[]):
        """ switch the connector at given node """
        obj = self & Q()
        *_, node = obj.resolve_path(path)
        if node.connector == Q.AND:
            node.connector = Q.OR
        elif node.connector == Q.OR:
            node.connector = Q.AND
        else:
            raise ValueError('invalid connector')
        return obj

    def get_field(self, path, model=None):
        """
        Get Field instance of rhs of end of given path.
        """
        if model is None:
            raise ValueError('model must be provided')

        *_, node = self.resolve_path(path)

        lhs = node[0].split('__')
        if lhs[-1] in models.Field.get_lookups():
            lhs = lhs[:-1]
        return model.get_field(lhs.join('__'))


class FKCacheBin(dict):
    """
    Mapping from PKs to model instances (or str() results)
    """
    def __init__(self, field):
        """
        field:
            ForeignKey field for which values shall be cached.
        """
        super().__init__(self)
        self.field_name = field.name
        self.attname = field.attname
        manager = field.related_model.objects
        if hasattr(manager, 'str_only'):
            self.related_queryset = manager.str_only()
        else:
            self.related_queryset = manager.all()

    def _get_row_pos(self, queryset):
        """
        Helper to get our values' position in row (for values list querysets)
        """
        if queryset._fields:
            names = queryset._fields
        else:
            # is empty tuple, so getting it all, cf. ValuesListIterable
            names = [
                *queryset.query.extra_select,
                *queryset.query.values_select,
                *queryset.query.annotation_select,
            ]
        return names.index(self.attname)

    def update_bin(self, queryset):
        """
        Checks queryset for missing related objects and retrieve them

        If the queryset is of the values list type, then just store the string
        representation, otherwise we store the related model instance.
        """
        if queryset._fields is None:
            # regular models-queryset
            get_fk = attrgetter(self.attname)
        else:
            # values list queryset
            try:
                pos = queryset.get_output_field_names().index(self.field_name)
            except ValueError:
                # field_name is not in list, try attname
                pos = queryset.get_output_field_names().index(self.attname)

            get_fk = itemgetter(pos)
            del pos

        missing = set()
        for i in queryset:
            fk = get_fk(i)
            if fk is None:
                continue
            if fk not in self:
                missing.add(fk)

        relqs = self.related_queryset.filter(pk__in=missing)

        if queryset._fields is None:
            for obj in relqs:
                self[obj.pk] = obj
        else:
            for obj in relqs:
                # for values lists convert to str right here
                self[obj.pk] = str(obj)

    def set_related_obj(self, obj):
        """
        Set the related FK object

        Assumes we were update properly, will raise KeyError otherwise.
        """
        if (pk := getattr(obj, self.attname)) is not None:
            setattr(obj, self.field_name, self[pk])


class FKCache(dict):
    """
    Container to hold multiple PK -> object mappings

    It can use the data is holds to set the foreign key relations of a queryset
    """
    def __init__(self, model, fk_fields=None):
        """
        Values cache for tables

        Parameters:
        model: model of table
        fk_fields:
            ForeignKey fields belonging to the model for which values shall be
            cached.
        """
        super().__init__(self)
        self.model = model
        # FIXME: maybe remove mode options, if not needed (good for testing?)
        if fk_fields is None:
            fk_fields = [i for i in model._meta.get_fields() if i.many_to_one]

        for i in fk_fields:
            self[i] = FKCacheBin(i)

    def bins(self):
        """ convenience method to get the collection of cache bins """
        return self.values()

    def get_map_value_row_fn(self, queryset):
        """
        Return a function to convert a row tuple where FKs are replaced with
        their cached object value, usually the str-representation.
        """
        # 1. Construct a row-length list of Nones and cache-bins so that cached
        # FK fields correspond to their cache bin.
        list_of_caches = []
        for i in queryset.get_output_field_names():
            try:
                fk_field = self.model._meta.get_field(i)
            except FieldDoesNotExist:
                list_of_caches.append(None)
            else:
                list_of_caches.append(self.get(fk_field, None))

        def map_row_fn(row):
            return [
                val if (cache is None or val is None) else cache[val]
                for cache, val in zip(list_of_caches, row)
            ]

        return map_row_fn

    def update_chunk(self, queryset):
        """
        Assign FK relations to objects in the given queryset

        This will evaluate an yet un-evaluated queryset.  It may incur
        additional DB queries to fetch data not yet cached.
        """
        # 1. identify and get missing FKs
        for i in self.bins():
            i.update_bin(queryset)

        # 2. update objects
        for obj in queryset:
            for i in self.bins():
                i.set_related_obj(obj)

    def update_values_list(self, queryset):
        """
        Generate updated value listing

        queryset:
            A value_list()-queryset

        Returns a new list to replace the original queryset
        """
        # 1. identify and get missing FKs
        for i in self.bins():
            i.update_bin(queryset)

        mapper = self.get_map_value_row_fn(queryset)
        return [mapper(i) for i in queryset]


class BaseIterable:
    """
    Alternative iterable implementation for large table data export.

    To be used by our Queryset.iterate().  These iterators are to be used only
    once.  Results are always ordered by primary key.
    """
    DEFAULT_CHUNK_SIZE = 20000

    def __init__(self, queryset, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.DEFAULT_CHUNK_SIZE
        elif chunk_size <= 0:
            raise ValueError('chunk size must be positive')
        self.queryset = queryset
        self.chunk_size = chunk_size
        self._it = None

    def debug(self):
        """ print state of the iterator """
        pprint(vars(self))
        if self._it is not None:
            print(f'state: {getgeneratorstate(self._it)}')
            pprint(getgeneratorlocals(self._it))

    def __iter__(self):
        # Attach the iterator as attribute to allow easier introspection.
        # Allow this to happen only once per object lifetime to reduce possible
        # confusion.
        if self._it is None:
            self._it = self._iter()
        else:
            raise RuntimeError('iter() only allowed once here')
        yield from self._it

    def _iter(self):
        """
        The iterator/generator yielding the data.

        Must be provided by implementing class.
        """
        raise NotImplementedError


class ModelIterable(BaseIterable):
    """ Iterable over regular model-based querysets """
    def __init__(self, queryset, chunk_size, cache):
        super().__init__(queryset, chunk_size)
        if cache is True:
            # auto-caching-mode
            # TODO: pick up fields from only()?
            cache = FKCache(self.queryset.model, fk_fields=None)
        self.cache = cache
        self.queryset = self.queryset.order_by('pk')

    def _iter(self):
        qs = self.queryset
        chunk_size = self.chunk_size
        cache = self.cache
        last_pk = 0
        while True:
            chunk = qs.filter(pk__gt=last_pk)[:chunk_size]

            if cache:
                # update in-place
                cache.update_chunk(chunk)

            yield from chunk

            if len(chunk) < chunk_size:
                # no further results
                break

            last_pk = chunk[len(chunk) - 1].pk

        # so debug() can display this if we're closed
        self._final_iter_vars = pformat(locals())


class ValuesListIterable(BaseIterable):
    """ Iterable over values-listing querysets """

    def __init__(self, queryset, chunk_size, cache):
        super().__init__(queryset, chunk_size)
        qs = self.queryset
        outnames = qs.get_output_field_names()
        hide_pk = False
        try:
            pk_pos = outnames.index(qs.model._meta.pk.name)
        except ValueError:
            if 'pk' in outnames:
                pk_pos = outnames.index('pk')
            else:
                # we have to get PK to make chunking work
                qs = qs.values_list('pk', *outnames)
                pk_pos = 0
                hide_pk = True

        if cache is True:
            # auto-caching-mode
            fk_fields = []
            for i in qs.model._meta.get_fields():
                if not i.many_to_one:
                    continue
                if i.name in outnames or i.attname in outnames:
                    fk_fields.append(i)
            cache = FKCache(qs.model, fk_fields=fk_fields)

        qs = qs.order_by('pk')

        self.queryset = qs
        self.cache = cache
        self.pk_pos = pk_pos
        self.hide_pk = hide_pk

    @staticmethod
    def _rm_pk(row):
        """ helper to remove PK from a list row """
        del row[0]  # PK is always first elem if we have to remove it
        return row

    def _iter(self):
        qs = self.queryset
        cache = self.cache
        chunk_size = self.chunk_size
        pk_pos = self.pk_pos
        hide_pk = self.hide_pk
        rm_pk = self._rm_pk

        last_pk = 0
        while True:
            chunk = qs.filter(pk__gt=last_pk)[:chunk_size]

            # For non-empty chunk get last PK before they are removed.  Must
            # also avoid negative indexing in no-cache case where chunk is
            # queryset, so calculate last row via length.
            if chunk_length := len(chunk):
                last_pk = chunk[chunk_length - 1][pk_pos]

            if cache:
                # chunk is replaced, type is list now (was tuple)
                chunk = cache.update_values_list(chunk)
                if hide_pk:
                    # rm PK from list
                    chunk = ((rm_pk(row) for row in chunk))
            elif hide_pk:
                # rm PK from tuple (get new tuple via slicing)
                chunk = ((row[slice(1, None)] for row in chunk))

            yield from chunk

            if chunk_length < chunk_size:
                # no further results
                break

        # so debug() can display this when we're closed
        self._final_iter_vars = pformat(locals())


class QuerySet(models.QuerySet):
    def __init__(self, *args, manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._avg_by = None
        self._avg_fields = None
        self._pre_annotation_clone = None
        self._rev_rel_count_fields = []
        self._manager = manager

    def peek(self):
        """ The original __repr__() method of Django querysets """
        return super().__repr__()

    def __repr__(self):
        """ Replacement that doesn't hit the DB """
        if self._result_cache is None:
            info = '(not evaluated)'
        else:
            info = f'length={len(self)}'
        return f'{type(self).__name__}/{self.model.__name__}:{info}'

    @classmethod
    def pd_type(cls, field):
        """
        Map Django field type to pandas data type
        """
        str_fields = (
            models.CharField,
            models.TextField,
            models.ForeignKey,
        )
        int_fields = (
            models.IntegerField,
            models.AutoField,
        )
        if isinstance(field, str_fields):
            dtype = str
        elif isinstance(field, int_fields):
            dtype = pandas.Int64Dtype()
        elif isinstance(field, models.BooleanField):
            dtype = bool
        elif isinstance(field, (models.DecimalField, models.FloatField)):
            dtype = float
        else:
            raise ValueError('Field type not supported: {}: {}'
                             ''.format(field, field.get_internal_type()))
        return dtype

    def as_dataframe(self, *fields, natural=False):
        """
        Convert to pandas dataframe

        :param: fields str: Only return columns with given field names.  If
                            this empty then all fields are returned.
        :param: natural bool: If true, then replace id/pk of foreign
                              relation with natural representation.
        """
        if self._avg_by:
            return self._as_dataframe_avg(*fields)

        if not fields:
            fields = self.model.get_fields().names

        _fields = OrderedDict()
        for i in ['id'] + list(fields):
            try:
                _fields[i] = self.model.get_field(i)
            except LookupError:
                if not hasattr(self.model, i):
                    raise ValueError(
                        'not the name of a field or model attribute: {}'
                        ''.format(i)
                    )
                # not a real field
                _fields[i] = None
        fields = _fields
        # real field names has at least 'id', so it's never empty:
        real_field_names = [i for i in fields if fields[i] is not None]
        del _fields

        # get transposed value list (for real fields):
        data = list(map(list, zip(*self.values_list(*real_field_names))))

        if len(self) == 0:
            # we're empty, and data == [] but we need it to be a list of empty
            # lists, the pandas calls below will then make a nice empty
            # DataFrame (with columns and index) for us
            data = [[] for _ in real_field_names]

        # map field name to list index (for real fields):
        fidx = dict((j, i) for i, j in enumerate(real_field_names))

        index = pandas.Index(data[fidx['id']], dtype=int, name='id')
        df = pandas.DataFrame([], index=index)

        for name, field in fields.items():
            if name == 'id':
                continue

            if field is None:
                dtype = None
                # not a field we can retrieve directly from the DB, have to
                # hope this works, can be expensive (extra DB queries)
                # E.g. 'pk', 'natural',
                # TODO: other examples?
                col_dat = (row.get_value_related(name) for row in self)
            else:
                dtype = self.pd_type(field)
                col_dat = data[fidx[name]]

            if field is not None and field.is_relation and natural:
                # replace pks with natural key, 1 extra DB query
                if self.is_curated():
                    manager = field.related_model.curated
                else:
                    manager = field.related_model.objects
                qs = manager.iterator()
                nat_dict = {i.pk: i.natural for i in qs}
                col_dat = map(
                    lambda pk: None if pk is None else nat_dict[pk],
                    col_dat
                )

            kwargs = dict(index=index)
            if field is not None and field.choices:
                col_dat = pandas.Categorical(col_dat)
            else:
                if dtype is str and not field.is_relation:
                    # None become empty str
                    # prevents 'None' string to enter df str columns
                    # (but not for foreign key columns)
                    col_dat = ('' if i is None else i for i in col_dat)
                if dtype == bool:
                    # don't pass dtype=bool to Series, this would make Nones
                    # be identified as False in the Series, but without dtype
                    # Series will auto-detect the type and store Nones as NAs
                    pass
                else:
                    kwargs['dtype'] = dtype

            df[name] = pandas.Series(col_dat, **kwargs)

        return df

    def _as_dataframe_avg(self, *fields):
        """
        Convert a QuerySet with ValuesIterable to a data frame

        The implementation depends on having average() called on the QuerySet.
        """
        if not fields:
            fields = self._avg_fields

        # getting pks here for index
        index = pandas.MultiIndex.from_tuples(
            self.values_list(*self._avg_by),
            names=self._avg_by
        )
        df = pandas.DataFrame([], index=index)

        for i in fields:
            col_dat = []
            for row in self:
                val = row[i]
                if isinstance(val, Decimal):
                    val = float(val)
                col_dat.append(val)
            df[i] = pandas.Series(col_dat, index=index)
        return df

    def _filter_or_exclude(self, negate, *args, **kwargs):
        """
        Handle natural lookups for filtering operations
        """
        self._pre_annotation_clone = None
        if hasattr(self.model, 'resolve_natural_lookups'):
            kwargs = self.model.resolve_natural_lookups(**kwargs)
        return super()._filter_or_exclude(negate, *args, **kwargs)

    def _values(self, *fields, **expressions):
        """
        Handle the 'natural' fields for value retrievals
        """
        if 'natural' in fields:
            fields = [i for i in fields if i != 'natural']
        return super()._values(*fields, **expressions)

    def get_field_stats(self, fieldname, natural=False):
        """
        Get basic descriptive stats from a single field/column

        Returns a dict: stats_type -> obj
        Returning an empty dict indicates some error
        """
        if fieldname in ['id', 'pk']:
            # as_dataframe('id') does not return anything too meaningful.  If
            # we wanted to return something here we need to treat id something
            # different that ordinary int fields.
            return {}

        qs = self
        if natural and fieldname in self.model.get_fields().names \
                and self.model._meta.get_field(fieldname).is_relation:
            if self._avg_by:
                # average() was called, implying values(), so we have dict
                # results and not model instances, so can't call select_related
                # which would raise us a TypeError.
                pass
            else:
                # speedup
                qs = qs.select_related(fieldname)

        col = qs.as_dataframe(fieldname, natural=natural)[fieldname]
        count_stats = col.value_counts(dropna=False).sort_index()
        ret = {
            'choice_counts': count_stats,
        }

        if is_numeric_dtype(col):
            ret['description'] = col.describe()
        else:
            if count_stats.count() == 1:
                # all values the same
                ret['uniform'] = count_stats.to_dict()

            if count_stats.max() < 2:
                # column values are unique
                ret['unique'] = count_stats.max()

            try:
                not_blank = count_stats.drop(index='')
            except KeyError:
                pass
            else:
                if not_blank.max() < 2:
                    # column unique except for empties
                    ret['unique_blank'] = {
                        'BLANK': count_stats[''],
                        'NOT_BLANK': not_blank.sum(),
                    }

        return ret

    def annotate_rev_rel_counts(self):
        """
        Add reverse relation count annotations
        """
        count_args = {}
        rels = self.model.get_related_objects()

        for i in rels:
            kwargs = dict(distinct=True)
            f = {}
            if self.is_curated():
                f = i.related_model.curated.get_curation_filter()
            if f:
                kwargs['filter'] = f
            name = i.related_model._meta.model_name + '__count'
            count_args[name] = models.Count(i.name, **kwargs)

        qs = self.annotate(**count_args)
        if count_args:
            qs._pre_annotation_clone = self.all()
            if qs._rev_rel_count_fields is None:
                qs._rev_rel_count_fields = []
            qs._rev_rel_count_fields += count_args.keys()
        log.debug(f'COUNT COLS: {count_args}')
        return qs

    def sum_rev_rel_counts(self):
        """
        Aggregate sums over reverse relation counts

        Returns a dict with {<model_name>__count__sum: int} per rev rel count
        annotation.  If no such annotation exist then an empty dict is
        returned.
        """
        args = [models.Sum(i) for i in self._rev_rel_count_fields or []]
        return self.aggregate(*args)

    def count(self):
        """
        Count that optimizes count annotations away

        This overriding method checks _pre_annotation_clone for presence of a
        clone made before a count column annotation is made.  Running the count
        query without such annotations is faster.  The validity of the
        resulting count depends on the stripped annotations not having any
        filter effect.  Also the clone needs to be reset when adding new
        filters.

        Hence, the general QuerySet build order should have filters first and
        the count annotations added last.
        """
        if self._pre_annotation_clone is None:
            return super().count()
        else:
            return self._pre_annotation_clone.count()

    def average(self, *avg_by, natural=True):
        """
        Average data of DecimalFields

        :param str avg_by: One or more field names, by which to sort and group
                           the data before taking averages for decimal values.
        :param bool natural: If True, then the averaged-by fields will be
                             populated with their natural key, otherwise with
                             the primary key.
        """
        self._pre_annotation_clone = None
        # TODO: average over fields in reverse related models
        # add group count

        # annotation kwargs:
        kwargs = {'avg_group_count': models.Count('id')}
        for i in self.model.get_average_fields():
            kwargs[i.name] = models.Avg(i.name)

        qs = self.values(*avg_by)

        qs._avg_by = avg_by
        qs._avg_fields = list(avg_by) + list(kwargs)
        if natural:
            qs._iterable_class = natural_values_iterable_factory(
                self.model,
                *avg_by,
            )

        # restrict groups to non-NULL members
        for i in avg_by:
            qs = qs.exclude(**{i: None})

        qs = qs.order_by(*avg_by).annotate(**kwargs)
        return qs

    def is_curated(self):
        """
        Tell if qeryset was created with the CurationManager
        """
        return hasattr(self._manager, 'get_curation_filter')

    def _clone(self):
        """
        Extent non-public _clone() to keep track of extra state
        """
        c = super()._clone()
        c._avg_by = self._avg_by
        c._avg_fields = self._avg_fields
        c._pre_annotation_clone = self._pre_annotation_clone
        c._rev_rel_count_fields = list(self._rev_rel_count_fields)
        c._manager = self._manager
        return c

    def save_csv(self, path, sep='\t'):
        table = table_factory(self.model)(data=self)
        count = 0
        with Path(path).open('w') as ofile:
            for values in table.as_values():
                row = []
                for val in values:
                    if val is None:
                        val = ''
                    else:
                        val = str(val)
                    row.append(val)

                ofile.write(sep.join(row) + '\n')
                count += 1
        print(f'Saved {count} rows to {ofile.name}')

    def get_output_field_names(self):
        """
        Get the selected field (or otherwise) names, of fetched rows output
        fields, in correct order.

        Only for querysets on which values() or values_list() was called.  Will
        raise ValueError if called with an incompatible queryset instance.

        Compare this to what NamedValuesListIterable does.
        """
        if self._fields is None:
            raise ValueError(
                'method shall only be called after values() or values_list()'
            )
        elif self._fields:
            return self._fields
        else:
            return [
                *self.query.extra_select,
                *self.query.values_select,
                *self.query.annotation_select,
            ]

    def iterate(self, chunk_size=None, cache=None):
        """
        Alternative iterator implementation for large table data export

        chunk_size:
            How much to get per DB query.  For values much below 1000 the cost
            of chunking becomes noticable.
        cache:
            An optional FKCache object to be used to populate FK related
            objects.  If this is None or an empty dict then no such cache will
            be used.  If True, then an FKCache will be used automatically.
        """
        if self._fields is None:
            # normal model-instance queryset
            return ModelIterable(self, chunk_size, cache)
        else:
            return ValuesListIterable(self, chunk_size, cache)
