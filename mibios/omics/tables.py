from django.db.models import IntegerField, Value
from django.db.models.functions import Cast, Replace
from django.utils.html import format_html

from django_tables2 import Column, Table

from mibios.glamr.utils import get_record_url

from . import get_sample_model
from .models import File, SampleTracking


class FileTable(Table):
    """
    Table of files, for internal display
    """
    file = Column(
        accessor='file_pipeline',
        verbose_name='File',
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
    modtime = Column(verbose_name='last modified')

    class Meta:
        model = File
        fields = ['file', 'file_local', 'file_globus', 'filetype',
                  'size', 'modtime']
        order_by = 'file'

    def render_file_local(self, value, record):
        return 'yes' if value else 'no'

    def render_file_globus(self, value, record):
        return 'yes' if value else 'no'

    def order_file(self, queryset, is_descending):
        """
        Order by sample id, numerically, ascending
        """
        qs = queryset.annotate(
            samp_id_s=Replace('sample__sample_id', Value('samp_'), Value('')),
            samp_id_i=Cast('samp_id_s', output_field=IntegerField()),
        )
        arg = 'samp_id_i'
        if is_descending:
            arg = '-' + arg
        qs = qs.order_by(arg)
        return qs, True


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
