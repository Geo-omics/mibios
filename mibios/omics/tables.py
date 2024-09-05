from django.utils.html import format_html

from django_tables2 import Column, Table

from mibios.glamr.utils import get_record_url

from . import get_sample_model
from .models import SampleTracking


class SampleTrackingTable(Table):
    sample__dataset__dataset_id = Column(
        verbose_name='Dataset',
        order_by=('sample__dataset__get_record_id_no', 'sample_id_num'),
    )
    sample__sample_id = Column(
        verbose_name='Sample',
        order_by='sample_id_num',
    )

    class Meta:
        # tracking flags to show:
        flags = [
            txt
            for name, txt
            in SampleTracking.Flag.choices
            if name != 'MD'  # skip "meta data loaded' which is always True
        ]

        model = get_sample_model()
        fields = [
            'sample__dataset__dataset_id',
            'sample__sample_id',
        ] + flags + [
            'sample__analysis_dir',
            'sample__read_count',
            'sample__reads_mapped_contigs',
            'sample__reads_mapped_genes',
        ]

    def render_sample__sample_id(self, record):
        return format_html(
            '<a href="{url}">{sample_id}</a>',
            url=get_record_url(record['sample']),
            sample_id=record['sample'].sample_id,
        )
