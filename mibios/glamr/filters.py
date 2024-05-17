from django.forms.widgets import CheckboxSelectMultiple

from django_filters import CharFilter, ChoiceFilter, DateFromToRangeFilter, \
        FilterSet, MultipleChoiceFilter
from django_filters.widgets import RangeWidget

from mibios.glamr.models import Dataset

from . import GREAT_LAKES


class AutoChoiceMixin:
    """
    Self-populating choices for filters

    Choices get cached at first access and will not get updated as the
    databases changes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    _choices = {}

    @classmethod
    def populate_choices(cls, model, field_name):
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

        cls._choices[(model, field_name)] = choices

    def ensure_extra(self):
        if 'choices' not in self.extra:
            if (self.model, self.field_name) not in self._choices:
                self.populate_choices(self.model, self.field_name)
            self.extra['choices'] = self._choices[(self.model, self.field_name)]  # noqa:E501

    @property
    def field(self):
        self.ensure_extra()
        return super().field


class WaterBodyFilter(AutoChoiceMixin, ChoiceFilter):
    lookup_expr = 'icontains'
    conjoined = False
    required = True

    @classmethod
    def populate_choices(cls, model, field_name):
        values = set()
        qs = model.objects.values_list('water_bodies', flat=True)
        for i in qs.distinct():
            for j in i.split(','):
                j = j.strip()
                if j:
                    values.add(j)
        values = values.difference(GREAT_LAKES)
        choices = [(i, i.capitalize()) for i in values]
        choices = sorted(choices, key=lambda x: x[1].casefold())
        choices = [(i, i) for i in GREAT_LAKES] + choices
        cls._choices[(model, field_name)] = choices


class SampleTypeFilter(AutoChoiceMixin, MultipleChoiceFilter):
    def ensure_extra(self):
        super().ensure_extra()
        self.extra['widget'] = CheckboxSelectMultiple()


class DatasetFilter(FilterSet):
    water_bodies = WaterBodyFilter()
    sample__sample_type = SampleTypeFilter(label='Sample type')
    sample_year = CharFilter(
        method='search_sample_date',
        label='Sample year (YYYY)',
    )
    sample__collection_timestamp = DateFromToRangeFilter(
        label="Sample Collection Date Range",
        widget=RangeWidget(attrs={'type': 'date'}),
    )

    class Meta:
        model = Dataset
        fields = [
            'water_bodies',
            'sample__sample_type',
            'sample_year',
            'sample__collection_timestamp',
        ]

    def search_sample_date(self, qs, name, value):
        return qs.filter(sample__collection_timestamp__year=value)

    def search_sample_locations(self, qs, name, value):
        return qs.filter(sample__geo_loc_name__icontains=value)
