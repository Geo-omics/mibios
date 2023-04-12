import django_filters

from mibios.glamr.models import Dataset


class DatasetFilter(django_filters.FilterSet):
    water_bodies = django_filters.CharFilter(
        label="Water bodies",
        lookup_expr='icontains',
    )
    scheme = django_filters.CharFilter(
        label="Description",
        lookup_expr='icontains',
    )
    material_type = django_filters.CharFilter(
        label="Material type",
        lookup_expr='icontains',
    )
    reference__authors = django_filters.CharFilter(
        label="Reference authors",
        lookup_expr='icontains',
    )
    reference__title = django_filters.CharFilter(
        label="Reference title",
        lookup_expr='icontains',
    )
    sample_location = django_filters.CharFilter(
        method='search_sample_locations',
        label='Sample location',
        # change label field to reflect what the filter name should be
    )

    SAMPLE_TYPES = (
        ('amplicon', 'Amplicon'),
        ('metagenome', 'Metagenome'),
        ('metatranscriptome', 'Metatranscriptome'),
    )
    sample__sample_type = django_filters.ChoiceFilter(
        label='Sample type',
        choices=SAMPLE_TYPES,
    )
    sample_year = django_filters.CharFilter(
        method='search_sample_date',
        label='Sample year (YYYY)',
    )

    sample__collection_timestamp = django_filters.DateFromToRangeFilter(
        label="Sample Collection Date Range",
        widget=django_filters.widgets.RangeWidget(attrs={'type': 'date'}),
    )

    class Meta:
        model = Dataset
        fields = [
            'water_bodies',
            'material_type',
            'scheme',
            'reference__title',
            'reference__authors',
            'sample_location',
            'sample__sample_type',
            'sample_year',
            'sample__collection_timestamp',
        ]

    def search_sample_date(self, qs, name, value):
        return qs.filter(sample__collection_timestamp__year=value)

    def search_sample_locations(self, qs, name, value):
        return qs.filter(sample__geo_loc_name__icontains=value)
