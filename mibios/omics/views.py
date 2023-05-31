from django.http import Http404, HttpResponse

from django_tables2 import SingleTableView

from . import get_sample_model
from .models import TaxonAbundance
from .tables import SampleStatusTable


def krona(request, sample_pk, stats_field):
    """
    Display Krona visualization for taxon abundance of one sample
    """
    # fields of TaxonAbundance that we allow for visualization
    ALLOWED_FIELD_NAMES = [
        'count_contig',
        'count_gene',
        'len_contig',
        'len_gene',
        'mean_fpkm_contig',
        'mean_fpkm_gene',
        'wmedian_fpkm_contig',
        'wmedian_fpkm_gene',
        'norm_reads_contig',
        'norm_reads_gene',
        'norm_frags_contig',
        'norm_frags_gene',
    ]

    Sample = get_sample_model()
    try:
        sample = Sample.objects.get(pk=sample_pk)
    except Sample.DoesNotExist:
        raise Http404('no such sample')

    if stats_field not in ALLOWED_FIELD_NAMES:
        raise Http404('bad stats field name')

    html = TaxonAbundance.objects.as_krona_html(sample, stats_field)
    if html is None:
        raise Http404('no abundance data for sample or error with krona')

    return HttpResponse(html)


class SampleStatusView(SingleTableView):
    template_name = 'omics/sample_status.html'
    table_class = SampleStatusTable
    model = get_sample_model()
    table_pagination = False
