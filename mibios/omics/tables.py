from django.utils.html import format_html

from django_tables2 import Column, Table

from mibios.glamr.utils import get_record_url

from . import get_sample_model
from .models import SampleTracking


class SampleTrackingTable(Table):
    sample__sample_id = Column(
        verbose_name='Sample',
        order_by='sample_id_num',
    )

    class Meta:
        model = get_sample_model()
        fields = [
            'sample__sample_id',
            # 'sample_id_num',
        ] + [txt for _, txt in SampleTracking.Flag.choices] + [
            'sample__analysis_dir',
            'sample__read_count',
            'sample__reads_mapped_contigs',
            'sample__reads_mapped_genes',
        ]

    def render_sample__sample_id(self, record):
        sample = record['sample']
        if record['private']:
            return format_html(
                '<span title="dataset private">{}</span>',
                sample.sample_id,
            )
        else:
            return format_html(
                '<a href="{url}">{sample_id}</a>',
                url=get_record_url(sample),
                sample_id=sample.sample_id,
            )
