import inspect
import sys

from django.forms.widgets import CheckboxSelectMultiple

from django_filters import CharFilter, ChoiceFilter, DateFromToRangeFilter, \
        FilterSet, MultipleChoiceFilter
from django_filters.widgets import RangeWidget

from mibios.glamr.models import Dataset, Sample
from mibios.umrad.models import UniRef100

from . import GREAT_LAKES


class AutoChoiceMixin:
    """
    Self-populating choices for filter components

    Choices get cached at first access and will not get updated as the
    databases changes.
    """
    _choices = {}
    widget_class = None

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
            qs = model1.objects.order_by(field_name1)
            qs = qs.values_list(field_name1, flat=True).distinct()
            choices = [(i, i) for i in qs]

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


class WaterBodyFiFi(AutoChoiceMixin, ChoiceFilter):
    lookup_expr = 'icontains'
    conjoined = False
    required = True

    def get_choices(cls, model, field_name):
        values = set()
        qs = model.objects.values_list('water_bodies', flat=True)
        for i in qs.distinct():
            for j in i.split(','):
                j = j.strip()
                if j:
                    values.add(j)
        values = values.difference(GREAT_LAKES)
        choices = [(i, i) for i in values]
        choices = sorted(choices, key=lambda x: x[1].casefold())
        return [(i, i) for i in GREAT_LAKES] + choices


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
    code = None


class DatasetFilter(StandardFilter):
    water_bodies = WaterBodyFiFi()
    sample__sample_type = SampleTypeFiFi(label='Sample type')
    sample_year = YearChoiceFiFi(
        method='add_year',
        label='Sample year (YYYY)',
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


class SampleFilter(StandardFilter):
    code = 'sa'

    class Meta:
        model = Sample
        exclude = []


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
