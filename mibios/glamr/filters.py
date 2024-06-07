import inspect
import sys

from django.db.models import QuerySet

from django.forms.widgets import CheckboxInput, CheckboxSelectMultiple

from django_filters import BooleanFilter, CharFilter, ChoiceFilter, \
    DateFromToRangeFilter, FilterSet, MultipleChoiceFilter, RangeFilter
from django_filters.widgets import RangeWidget

from mibios.glamr.models import Dataset, Reference, Sample
from mibios.umrad.models import UniRef100


class AutoChoiceMixin:
    """
    Self-populating choices for filter components

    Choices get cached at first access and will not get updated as the
    databases changes.

    The constructor supports these additional keyword arguments:

    sort_key:
        A callable, will be passed to sorted() to override the normal ordering
        of the field by the DB.
    sep:
        separator on which to split listing.  White-space trimming is also done
        and the resulting items are ordered alphabetically case-insentitive by
        default, but sort_key can override this.
    blank_value:
        How blank values are stored at the DB.  Defaults to the empty string.
    """
    _choices = {}
    widget_class = None
    DEFAULT_BLANK_LABEL = '<blank>'
    DEFAULT_BLANK_VALUE = ''

    def __init__(self, *args, sort_key=None, sep=None, **kwargs):
        if sep:
            kwargs.setdefault('lookup_expr', 'icontains')
        self.blank_value = kwargs.pop('blank_value', self.DEFAULT_BLANK_VALUE)
        # don't pass null_label or django_filters will put blank choice on top
        self.blank_label = kwargs.pop('null_label', self.DEFAULT_BLANK_LABEL)
        super().__init__(*args, **kwargs)
        self.sort_key = sort_key
        self.sep = sep

    @classmethod
    def _get_choices(cls, model, field_name, instance):
        """
        populate and retrieve choices from class-level cache

        Choices are kept per model/field pair so that a filter class can
        be used in different filtersets.
        """
        if (model, field_name) not in cls._choices:
            cls._choices[(model, field_name)] = \
                instance.get_choices(model, field_name)
        return cls._choices[(model, field_name)]

    def get_choices(self, model, field_name):
        """
        Build the list of choices.

        Returns a list of pairs of internal and display values.  Inheriting
        classes can overwrite this method.
        """
        if '__' in field_name:
            # follow that relation
            field = model.get_field(field_name)
            model1 = field.model
            field_name1 = field.name
        else:
            model1 = model
            field_name1 = field_name

        if choices := model1._meta.get_field(field_name1).choices:
            # prefer using declared choices
            pass
        else:
            # retrieve possible values from database
            qs = model1.objects.order_by(field_name1)
            qs = qs.values_list(field_name1, flat=True).distinct()

            # sort out blanks (empty string)
            # this is entriely separate from django_filters' handling of null
            # choice values, which we don't touch
            values = []
            have_blank = False
            for i in qs:
                if i == self.blank_value:
                    have_blank = True
                else:
                    values.append(i)

            # process lists if needed
            if self.sep:
                values1 = set()
                for txt in values:
                    for item in txt.split(self.sep):
                        item = item.strip()
                        if item:
                            values1.add(item)
                if self.sort_key is None:
                    values = sorted(values1, key=lambda x: x.casefold())
                else:
                    values = values1

            # override order is needed
            if self.sort_key is not None:
                values = sorted(values, key=self.sort_key)

            choices = [(i, i) for i in values]

            # put any blank last
            if have_blank:
                choices.append((self.null_value, self.blank_label))

        return choices

    def ensure_extra(self):
        """ Populate the extra attribute with things to be passed to the form
        field """
        if 'choices' not in self.extra:
            self.extra['choices'] = \
                self._get_choices(self.model, self.field_name, self)

        if 'widget' not in self.extra and self.widget_class is not None:
            self.extra['widget'] = self.widget_class()

    @property
    def field(self):
        self.ensure_extra()
        return super().field

    def filter(self, qs, value):
        # super()'s filter() does null value lookup only to None, not out more
        # common empty string.  We also use the 'exact' lookup, unsure if any
        # other lookup ever makes any sense
        if value == self.null_value and self.blank_value is not None:
            # '' needs exact lookup; super() can handle None
            kw = {f'{self.field_name}__exact': self.blank_value}
            qs = self.get_method(qs)(**kw)
            return qs.distinct() if self.distinct else qs
        else:
            return super().filter(qs, value)


class OkOnlyFiFi(BooleanFilter):
    """
    The Ok-Only filter presents a checkbox.  If checked we return objects for
    which the field is True.  If left unchecked, then no filter is applied.

    Target use are the _ok fields of Sample.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('widget', CheckboxInput())
        super().__init__(*args, **kwargs)

    def filter(self, qs, value):
        if value is False:
            # unchecked, no filtering
            return qs
        else:
            return super().filter(qs, value)


class ChoiceFiFi(AutoChoiceMixin, ChoiceFilter):
    pass


class SampleTypeFiFi(AutoChoiceMixin, MultipleChoiceFilter):
    widget_class = CheckboxSelectMultiple


class YearChoiceFiFi(AutoChoiceMixin, ChoiceFilter):
    def get_choices(cls, model, field_name):
        qs = (Sample.objects
              .exclude(collection_timestamp=None)
              .order_by('collection_timestamp__year')
              .values_list('collection_timestamp__year', flat=True)
              .distinct()
              )
        return [(i, i) for i in qs]


class StandardFilter(FilterSet):
    """ Abstract class for filters that appear in the 'standard' section of the
    advanced search """
    code = None
    """ The code: goes into URL so the view knows the filter to use """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied_filters = []

    def for_display(self):
        """
        Get display items for a bound+cleaned filter / form / set
        """
        ret = []
        for name in self.applied_filters:
            fi = self.filters[name]
            label = fi._label
            value = self.form.cleaned_data[name]
            if isinstance(value, slice):
                # date range or similar
                value = f'{value.start} to {value.stop}'
            elif isinstance(value, list):
                # multi-valued choices or so
                value = ', '.join((str(i) for i in value))
            elif hasattr(fi, 'null_value') and value == fi.null_value:
                # probably "null", try getting blank label from AutoChoiceMixin
                value = getattr(self.filters[name], 'blank_label', value)
            else:
                # assume str() will work just fine
                pass

            ret.append((label, value))
        return ret

    def filter_queryset(self, queryset):
        """
        Filter the queryset based on the form's cleaned_data

        This replaces super()'s method to also keep track of which filters
        actually got applied.
        """
        for name, value in self.form.cleaned_data.items():
            old_queryset = queryset
            queryset = self.filters[name].filter(queryset, value)
            if not isinstance(queryset, QuerySet):
                raise TypeError(
                    f'Expected {type(self).__name__}{name} to return a '
                    f'QuerySet, but got a {type(queryset).__name__} instead.'
                )
            if queryset is not old_queryset:
                # filter applied non-trivially
                self.applied_filters.append(name)
        return queryset

    @classmethod
    def filter_for_field(cls, field, field_name, lookup_expr=None):
        ret = super().filter_for_field(field, field_name,
                                       lookup_expr=lookup_expr)
        return ret


class DatasetFilter(StandardFilter):
    water_bodies = ChoiceFiFi(sep=',')
    sample__sample_type = SampleTypeFiFi(label='Sample type')
    sample_year = YearChoiceFiFi(
        method='add_year',
        label='Sample year',
    )
    sample__collection_timestamp = DateFromToRangeFilter(
        label="Sample Collection Date Range",
        widget=RangeWidget(attrs={'type': 'date'}),
    )

    code = 'da'

    class Meta:
        model = Dataset
        fields = [
            'water_bodies',
            'sample__sample_type',
            'sample_year',
            'sample__collection_timestamp',
        ]

    def add_year(self, qs, name, value):
        return qs.filter(sample__collection_timestamp__year=value)


class ReferenceFilter(StandardFilter):
    last_author = ChoiceFiFi()
    year = ChoiceFiFi(label='Year of publication')
    key_words = ChoiceFiFi(sep=',', label='Keywords')
    publication = ChoiceFiFi(label='Journal')

    code = 'pub'

    class Meta:
        model = Reference
        fields = ['last_author', 'year', 'key_words', 'publication']


class SampleFilter(StandardFilter):
    geo_loc_name = ChoiceFiFi()
    year = YearChoiceFiFi(
        method='add_year',
        label='Year',
    )
    collection_timestamp = DateFromToRangeFilter(
        label="Sample Collection Date Range",
        widget=RangeWidget(attrs={'type': 'date'}),
    )

    code = 'sa'

    class Meta:
        model = Sample
        fields = [
            'geo_loc_name', 'tax_abund_ok', 'year', 'collection_timestamp',
            'sample_type', 'amplicon_target', 'fwd_primer', 'rev_primer',
            'microcystis_count',
        ]

    @classmethod
    def filter_for_field(cls, field, field_name, lookup_expr=None):
        if field_name.endswith('_ok'):
            return OkOnlyFiFi(field_name=field_name, lookup_expr=lookup_expr)

        itype = field.get_internal_type()
        if 'Integer' in itype or 'Decimal' in itype:
            return RangeFilter(
                field_name=field_name,
                lookup_expr=lookup_expr,
            )

        if itype in ['Textfield', 'CharField'] and lookup_expr is None:
            lookup_expr = 'icontains'

        # last resort
        return super().filter_for_field(field, field_name,
                                        lookup_expr=lookup_expr)

    def add_year(self, qs, name, value):
        return qs.filter(collection_timestamp__year=value)


class UniRef90Filter(FilterSet):
    uniref90 = CharFilter(label='UniRef90 ID')

    code = 'u9'

    class Meta:
        model = UniRef100
        fields = ['uniref90']


class UniRef100Filter(FilterSet):
    accession = CharFilter(label='UniRef100 ID')

    code = 'u1'

    class Meta:
        model = UniRef100
        fields = ['accession']


def _compile_filter_registry():
    reg = {}
    filters = inspect.getmembers(
        sys.modules[__name__],
        lambda x: inspect.isclass(x)
        and x.__module__ == __name__
        and issubclass(x, FilterSet)
        and hasattr(x, 'code')
    )
    for name, cls in filters:
        if name == 'StandardFilter':
            continue
        if cls.code is None:
            raise ValueError(f'filter registry code not set: {cls}')
        if cls.code in reg:
            raise ValueError(f'Class {cls} has duplicate code: {cls.code}')
        reg[cls.code] = cls
    return reg


# this should go after all class declarations
filter_registry = _compile_filter_registry()
""" a map from sorted field names / accessors to filter set class for
auto-picking the right class from a GET querystring """

standard_filters = StandardFilter.__subclasses__()
""" filters to be included on the adv search page """
