from django.db.models.functions import Coalesce
from django.utils.html import format_html

from django_tables2 import A, Column, Table

from mibios.glamr.utils import get_record_url

from . import get_sample_model
from .models import File, SampleTracking


class FileTable(Table):
    """
    Table of files, for internal display
    """
    download_url = Column(
        verbose_name='File',
        order_by='sample__dataset, sample',
        empty_values=(),  # triggers render_FOO()
    )
    is_public = Column(
        accessor=A('public'),
        verbose_name='Public?',
        empty_values=(),
    )

    class Meta:
        model = File
        fields = ['download_url', 'is_public', 'filetype', 'size', 'modtime']

    def render_download_url(self, value, record):
        if record.public:
            path = record.public.relative_to(File.get_public_prefix())
            if value:
                return format_html('<a href="{}">{}</a>', value, path)
            else:
                return f'{path} (unavailable)'
        else:
            path = record.path.relative_to(File.get_path_prefix())
            return str(path)

    def order_download_url(self, queryset, is_descending):
        qs = queryset.annotate(path0=Coalesce('public', 'path'))
        qs = qs.order_by('sample', ('-' if is_descending else '') + 'path0')
        return (qs, True)

    def render_is_public(self, value, record):
        return 'yes' if record.public else 'no'


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
