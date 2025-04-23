from collections import defaultdict

from django.urls import reverse
from django_tables2 import A, Column, Table

from .models import Sample


def get_samples_url(record):
    """ linkify helper for DatasetTable """
    # FIXME: any less ad-hoc solution (reverse django_filter)?
    return reverse('sample_list') + f'?dataset={record.pk}'


class ASVAbundanceTable(Table):
    sample = Column(linkify=True)
    asv = Column(
        verbose_name='ASV',
        linkify=('asv_detail', {'asvnum': A('asv.asv_number')}),
    )
    count = Column()
    relabund = Column()


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.types = None

    def render_sample_type(self, table, record):
        if self.types is None:
            qs = Sample.objects.filter(dataset__in=table.data)
            qs = qs.values_list('dataset__pk', 'sample_type').distinct()
            self.types = defaultdict(list)
            for pk, sample_type in qs:
                self.types[pk].append(sample_type)

        return ', '.join(sorted(self.types[record.pk]))


class SampleTable(Table):
    dataset = Column(linkify=True)
    label = Column(linkify=True)
    sample_type = Column()
    amplicon_target = Column()
    source_material = Column()
    host = Column(linkify=True)
