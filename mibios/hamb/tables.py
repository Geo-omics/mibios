from collections import defaultdict

from django.urls import reverse
from django_tables2 import A, Column, Table

from .models import Host, Sample


def get_samples_url(record):
    """ linkify helper for DatasetTable """
    # FIXME: any less ad-hoc solution (reverse django_filter)?
    return reverse('sample_list') + f'?dataset={record.pk}'


class ASVTable(Table):
    accession = Column(
        linkify=('asv_detail', {'asvnum': A('asv_number')})
    )
    taxon__name = Column(
        linkify=('taxon_detail', {'taxid': A('taxon__taxid')})
    )


class ASVAbundanceTable(Table):
    sample = Column(linkify=True)
    asv = Column(
        verbose_name='ASV',
        linkify=('asv_detail', {'asvnum': A('asv.asv_number')}),
    )
    asv__taxon = Column(
        linkify=('taxon_detail', {'taxid': A('asv__taxon__taxid')}),
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


class HostTable(Table):
    dataset = Column(empty_values=())
    sample_count = Column(
        verbose_name='Samples',
        linkify=('host_sample_list', {'pk': A('pk')}),
    )
    label = Column(
        verbose_name='Host',
        linkify=True,
    )

    class Meta:
        model = Host
        sequence = ['dataset', 'sample_count', '...']
        exclude = ('id', 'description')

    def render_dataset(self, record):
        datasets = record.sample_set.values_list('dataset__label', flat=True)
        datasets = datasets.distinct()
        return ', '.join(datasets)


class SampleTable(Table):
    dataset = Column(linkify=True)
    label = Column(linkify=True)
    sample_type = Column()
    amplicon_target = Column()
    source_material = Column()
    host = Column(linkify=True)
