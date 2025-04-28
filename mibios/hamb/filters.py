from django_filters import (
    ChoiceFilter, FilterSet, ModelChoiceFilter
)

from mibios.omics.models import ASV, ASVAbundance
from .models import Dataset, Sample


BLANK = '_EMPTY_'


class AutoChoiceFilter(ChoiceFilter):
    """
    Choice filter with choices taken from DB

    Similar to AllValuesFilter but only queries DB once at beginning and
    handles empty values.
    """
    BLANKS = ('', None)  # blank/empty DB values
    BLANK_CHOICE = ('_EMPTY_', '<blank>')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    _auto_choices = {}
    _blank_values = {}

    @classmethod
    def get_auto_choices(cls, model, field_name):
        UNSET = object()
        if (model, field_name) not in cls._auto_choices:
            qs = model._default_manager.distinct()
            qs = qs.order_by(field_name)
            qs = qs.values_list(field_name, flat=True)
            choices = []
            blank_value = UNSET
            for i in qs:
                if i in cls.BLANKS:
                    blank_value = i
                else:
                    choices.append((i, i))

            if blank_value is not UNSET:
                choices.append(cls.BLANK_CHOICE)
                cls._blank_values[(model, field_name)] = blank_value

            cls._auto_choices[(model, field_name)] = choices
        return cls._auto_choices[(model, field_name)]

    # cf. AllValuesField, hooks into field()
    @property
    def field(self):
        self.extra['choices'] = \
            self.get_auto_choices(self.model, self.field_name)
        return super().field

    # cf. django_filter docs
    def filter(self, qs, value):
        if value != self.BLANK_CHOICE[0]:
            return super().filter(qs, value)

        blank_value = \
            self._blank_values.get((self.model, self.field_name), None)
        f = {f'{self.field_name}__{self.lookup_expr}': blank_value}
        qs = self.get_method(qs)(**f)
        return qs.distinct() if self.distinct else qs


def get_choices(model, field_name):
    """ helper to generate choices """
    qs = model.objects.values_list(field_name, flat=True)
    qs = qs.order_by(field_name).distinct()
    return tuple((
        (i, i) if i else (BLANK, '<blank>')
        for i in qs
    ))


class ASVFilter(FilterSet):
    class Meta:
        model = ASV
        fields = ['taxon__taxid']


class ASVAbundanceFilter(FilterSet):
    class Meta:
        model = ASVAbundance
        fields = ['asv__accession']


class DatasetFilter(FilterSet):
    sample_type = ChoiceFilter(
        field_name='sample__sample_type',
        label='sample type',
        choices=Sample.SAMPLE_TYPES_CHOICES,
    )

    class Meta:
        model = Dataset
        fields = ['sample_type']


class SampleFilter(FilterSet):
    dataset = ModelChoiceFilter(queryset=Dataset.objects.all())
    sample_type = ChoiceFilter(
        choices=Sample.SAMPLE_TYPES_CHOICES,
    )
    source_material = AutoChoiceFilter()
    control = AutoChoiceFilter()
    amplicon_target = AutoChoiceFilter()

    class Meta:
        model = Sample
        fields = [
            'dataset', 'sample_type', 'source_material', 'control',
            'amplicon_target',
        ]
