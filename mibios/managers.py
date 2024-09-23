from django.apps import apps
from django.db.models.manager import BaseManager as DjangoBaseManager

from .query import Q, QuerySet


class BaseManager(DjangoBaseManager):
    def get_by_natural_key(self, key):
        return self.get(**self.model.natural_lookup(key))

    def get_queryset(self):
        """
        Return a new QuerySet object.

        This overrides method in Django's BaseManager to give a reference to
        itself to the returned QuerySet.  Hence, we must use the
        mibios.QuerySet, Django's original QuerySet constructor will not
        understand the extra argument.
        """
        return self._queryset_class(model=self.model, using=self._db,
                                    hints=self._hints, manager=self)


class CurationBaseManager(BaseManager):
    """
    Manager to implement publishable vs. hidden data
    """
    filter = None
    excludes = None

    EXCLUDE_TAG = 'exclude'

    def __init__(self, *args, filter={}, excludes=[], **kwargs):
        super().__init__(*args, **kwargs)
        # Setting base filters, must be completed after all the managers are
        # fully initialized but that won't happen until the models are set up
        self.base_filter = filter.copy()
        self.base_excludes = []
        for i in excludes:
            self.base_excludes.append(i.copy())

    def ensure_filter_setup(self):
        """
        Follow foreign key relations recursively to set up filters

        TODO: handle cycles
        """
        if self.filter is not None and self.excludes is not None:
            return

        self.filter = self.base_filter.copy()
        self.excludes = []

        # tag filtering:
        for i in self.model.get_fields(with_m2m=True).fields:
            if i.related_model is apps.get_model('mibios.TagNote'):
                self.excludes.append({i.name + '__tag': self.EXCLUDE_TAG})

        self.excludes += [i.copy() for i in self.base_excludes]

        for i in self.model.get_fields().fields:
            if i.is_relation and (i.many_to_one or i.one_to_one):
                # is a foreign key
                other = i.related_model.curated
                prefix = i.name + '__'
                other.ensure_filter_setup()
                for k, v in other.filter.items():
                    self.filter[prefix + k] = v
                for i in other.excludes:
                    e = {prefix + k: v for k, v in i.items()}
                    if e:
                        self.excludes.append(e)

    def get_queryset(self):
        self.ensure_filter_setup()
        qs = super().get_queryset()

        if self.filter is not None:
            qs = qs.filter(**self.filter)
        for i in self.excludes:
            qs = qs.exclude(**i)

        return qs

    def get_curation_filter(self, prefix=None):
        """
        Return the curation filter/exclude as Q object

        :param str prefix:
            Accessor prefix.  Use this when the target model is related to one
            with a CurationManager but does not have one itself.  Then the
            prefix is the accessor to the related model without the foreign
            field itself, e.g. to effectively curation-filter
            mibios_seq.Abundance, which does not have a CurationManager, we
            need to filter on curated Sequencing:

            f = Sequencing.get_curation_filter(prefix='abundance')
            OTU.objects.annotate(sum=Sum('abundance__count', filter=f))

            Then the sum is only taken over curated sequencings.

        A Q return value can be used for the filter keyword in calls to
        Aggregate() and friends, e.g. Count().  A None return value can be used
        to determine that no such filter needs to be applied.

        The model name (and prefix if any) will be added to the lookup lhs.
        """
        self.ensure_filter_setup()
        if prefix:
            if not prefix.endswith('__'):
                prefix += '__'
        else:
            prefix = ''

        prefix += self.model._meta.model_name + '__'
        f = {prefix + k: v for k, v in self.filter.items()}
        e = [
            ~Q(**{prefix + k: v for k, v in i.items()})
            for i in self.excludes if i
        ]
        q = Q(*e, **f)
        if q:
            return q
        else:
            return None


class Manager(BaseManager.from_queryset(QuerySet)):
    pass


class CurationManager(CurationBaseManager.from_queryset(QuerySet)):
    pass
