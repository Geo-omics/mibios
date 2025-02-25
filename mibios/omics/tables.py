from django.utils.html import format_html

from django_tables2 import Column, Table

from mibios.glamr.utils import get_record_url

from . import get_sample_model
from .models import File, SampleTracking


class FileTable(Table):
    """
    Table of files, for internal display
    """
    file_pipeline = Column(
        verbose_name='File',
        order_by='sample__dataset, sample, file_pipeline',
    )
    file_local = Column(
        verbose_name='direct download',
        linkify=lambda value: value.url if value else None,
        empty_values=(),  # triggers render_FOO()
    )
    file_globus = Column(
        verbose_name='via Globus',
        linkify=lambda value: value.url if value else None,
        empty_values=(),  # triggers render_FOO()
    )

    class Meta:
        model = File
        fields = ['file_pipeline', 'file_local', 'file_globus', 'filetype',
                  'size', 'modtime']

    def render_file_local(self, value, record):
        return 'yes' if value else 'no'

    def render_file_globus(self, value, record):
        return 'yes' if value else 'no'


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
