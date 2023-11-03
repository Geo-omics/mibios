from django_tables2 import A, Column, Table

from . import get_sample_model


class SampleStatusTable(Table):
    sample_id = Column(linkify=('sample', [A('pk')]))

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
            'func_abund_ok',
            'comp_abund_ok',
            'binning_ok',
            'checkm_ok',
            'genes_ok',
            'proteins_ok',
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
