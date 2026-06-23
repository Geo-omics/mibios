from functools import partial

from django.db.models import IntegerField, Value
from django.db.models.functions import Cast, Replace
from django.utils.html import format_html, urlencode
from django.urls import reverse

from django_tables2 import Column, Table
from django_tables2.tables import DeclarativeColumnsMetaclass

from mibios.glamr.utils import get_record_url

from . import get_sample_model, get_dataset_model
from .models import DataTracking, File, SeqSample, SampleTracking


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


class TrackingColumnsMetaclass(DeclarativeColumnsMetaclass):
    """
    Metaclass to make a column for each tracking flag in certain order and also
    each sampletype.
    """

    _base_url = None
    """ base url for processed-sample-group links """

    @staticmethod
    def ds_tracking_url(flag_value, record, value):
        """
        linkify function for the flag columns

        Link to the group of samples belonging to a dataset that have data
        loaded according to the given tracking flag.

        This is a static method as classmethod and partialmethod doesn't seem
        to work together.
        """
        if value:
            qstr = urlencode({
                'filter-dataset__dataset_id': record.dataset_id,
                'filter-seqsample__tracking__flag': flag_value,
            })
            if not TrackingColumnsMetaclass._base_url:
                # base url gets cached
                TrackingColumnsMetaclass._base_url = \
                    reverse('filter_result', args=('sample',))
            return TrackingColumnsMetaclass._base_url + '?' + qstr
        else:
            return None  # don't link zeros

    @staticmethod
    def ds_sampletype_url(sampletype, record, value):
        """
        linkify function for the sample type columns

        Link to the group of seqsamples belonging to a dataset with given
        sample type.

        This is a static method as classmethod and partialmethod doesn't seem
        to work together.
        """
        if value:
            qstr = urlencode({
                'filter-dataset__dataset_id': record.dataset_id,
                'filter-seqsample__sample_type': sampletype,
            })
            if not TrackingColumnsMetaclass._base_url:
                # base url gets cached
                TrackingColumnsMetaclass._base_url = \
                    reverse('filter_result', args=('sample',))
            return TrackingColumnsMetaclass._base_url + '?' + qstr
        else:
            return None  # don't link zeros

    @classmethod
    def column_sum(mcs, bound_column, table):
        """
        Calculate sum of column for a footer row

        Assumes table data is a list (or evaluated queryset) of objects
        """
        return sum(getattr(obj, bound_column.name) for obj in table.data)

    def __new__(mcs, name, bases, attrs):
        flags = [DataTracking.Flag.METADATA, DataTracking.Flag.PIPELINE]  # these first
        flags += [i for i in DataTracking.Flag if i not in flags]

        for i in SeqSample.Type.values:
            linkify = partial(mcs.ds_sampletype_url, i)
            attrs[i] = Column(
                verbose_name=mcs.split_sample_type(i),
                linkify=linkify,
                default=0,
                footer=mcs.column_sum,
            )

        for flag in flags:
            linkify = partial(mcs.ds_tracking_url, flag.value)
            attrs[flag.value] = Column(
                verbose_name=flag.label,
                linkify=linkify,
                default=0,
                footer=mcs.column_sum,
            )

        return super().__new__(mcs, name, bases, attrs)

    @classmethod
    def split_sample_type(mcs, value):
        """ helper to make the column headers take up less space """
        value = value.replace('_', ' ')
        if value.startswith('meta'):
            value = 'meta ' + value.removeprefix('meta')
        return value

class DatasetTrackingTable(Table, metaclass=TrackingColumnsMetaclass):
    dataset_id = Column(linkify=True, footer='Totals:', order_by='pk')
    num_biosample = Column(footer=TrackingColumnsMetaclass.column_sum)
    total = Column(footer=TrackingColumnsMetaclass.column_sum)

    class Meta:
        model = get_dataset_model()
        fields = [
            'dataset_id',
            'private',
        ]


class SampleTrackingTable(Table):
    sample__parent__dataset__dataset_id = Column(
        verbose_name='Dataset',
        order_by=('parent__dataset__get_record_id_no', 'sample_id_num'),
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
            'sample__parent__dataset__dataset_id',
            'sample__sample_id',
        ] + flags + [
            'sample__analysis_dir',
            'sample__read_count',
            'sample__reads_mapped_contigs',
            'sample__reads_mapped_genes',
        ]
        order_by = ('sample__sample_id',)

    def render_sample__parent__dataset__dataset_id(self, record):
        return format_html(
            '<a href="{url}">{dataset_id}</a>',
            url=get_record_url(record['sample'].parent.dataset),
            dataset_id=record['sample'].parent.dataset.dataset_id,
        )

    def render_sample__sample_id(self, record):
        return format_html(
            '<a href="{url}">{sample_id}</a>',
            url=get_record_url(record['sample']),
            sample_id=record['sample'].sample_id,
        )
