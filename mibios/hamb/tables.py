from collections import defaultdict

from django_tables2 import Column, Table

from .models import Sample


def get_samples_url(record):
    """ linkify helper for DatasetTable """
    # TODO
    return '/xx'


class DatasetTable(Table):
    label = Column(linkify=True)
    sample_count = Column(
        verbose_name='Samples',
        linkify=get_samples_url,
    )
    sample_type = Column(
        empty_values=(),
        orderable=False,
    )

    class Meta:
        # model = Dataset
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.types = None

    def get_queryset(self):
        qs = super().get_queryset()
        return qs

    def render_sample_type(self, table, record):
        if self.types is None:
            qs = Sample.objects.filter(dataset__in=table.data)
            qs = qs.values_list('dataset__pk', 'sample_type').distinct()
            self.types = defaultdict(list)
            for pk, sample_type in qs:
                self.types[pk].append(sample_type)

        return ', '.join(sorted(self.types[record.pk]))


class SampleTable(Table):
    label = Column(linkify=True)
    host = Column(linkify=True)

    class Meta:
        # model = Dataset
        ...

    def get_queryset(self):
        qs = super().get_queryset()
        return qs
