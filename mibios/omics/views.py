from django.http import Http404, HttpResponse

from django_tables2 import SingleTableView

from . import get_sample_model
from .models import TaxonAbundance
from .tables import SampleStatusTable


def krona(request, samp_no):
    """
    Display Krona visualization for taxon abundance of one sample
    """
    Sample = get_sample_model()
    try:
        sample = Sample.objects.get(sample_id=f'samp_{samp_no}')
    except Sample.DoesNotExist:
        raise Http404('no such sample')

    html = TaxonAbundance.objects.as_krona_html(sample)
    if html is None:
        raise Http404('no abundance data for sample or error with krona')

    return HttpResponse(html)


class SampleStatusView(SingleTableView):
    template_name = 'omics/sample_status.html'
    table_class = SampleStatusTable
    model = get_sample_model()
    table_pagination = False
