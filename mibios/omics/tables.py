from django.utils.functional import cached_property
from django.utils.html import format_html

from django_tables2 import Table

from . import get_dataset_model, get_sample_model


class SampleStatusTable(Table):

    class Meta:
        model = get_sample_model()
        fields = [
            'sample_id',
            'meta_data_loaded',
            'metag_pipeline_reg',
            'contig_fasta_loaded',
            'contig_abundance_loaded',
            'contig_lca_loaded',
            'gene_alignments_loaded',
            'read_abundance_loaded',
            'tax_abund_ok',
            # 'func_abund_ok',
            # 'comp_abund_ok',
            # 'binning_ok',
            # 'checkm_ok',
            # 'genes_ok',
            # 'proteins_ok',
            'analysis_dir',
            'read_count',
            'reads_mapped_contigs',
            'reads_mapped_genes',
        ]
        order_by = 'sample_id'

    def order_sample_id(self, qs, is_descending):
        if is_descending:
            sign = '-'
        else:
            sign = ''
        qs = qs.order_by(f'{sign}pk')
        return (qs, True)

    @cached_property
    def dataset_is_private(self):
        return dict(get_dataset_model().loader.values_list('pk', 'private'))

    def render_sample_id(self, record):
        if self.dataset_is_private[record.dataset_id]:
            return format_html(
                '<span title="dataset private">{}</span>',
                record.sample_id,
            )
        else:
            return format_html(
                '<a href="{url}">{sample_id}</a>',
                url=record.get_record_url(record),
                sample_id=record.sample_id,
            )
