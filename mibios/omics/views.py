from collections import defaultdict
from itertools import groupby

from django.conf import settings
from django.http import Http404, HttpResponse

from django_tables2 import SingleTableView

from mibios.views import StaffLoginRequiredMixin
from . import get_dataset_model, get_sample_model
from .models import SampleTracking, TaxonAbundance
from .tables import SampleTrackingTable


class RequiredSettingsMixin:
    """
    Require one or more boolean settings to be True

    Implementing classes must set the "required_settings" attribute to a string
    of list of strings naming one or more settings.  All of these settings must
    be True for the view to be served.  Otherwise, or if the required_settings
    is not set correctly, a 404 response is returned.
    """
    required_settings = None

    def setup(self, *args, **kwargs):
        if isinstance(self.required_settings, str):
            settings_attrs = [self.required_settings]
        else:
            settings_attrs = self.required_settings

        if not settings_attrs:
            raise Http404(
                'required_settings is not configured, must be a settings '
                'attribute name or list of them'
            )

        for i in settings_attrs:
            if getattr(settings, i, False) is not True:
                raise Http404(f'settings.{i} is not True')

        super().setup(*args, **kwargs)


def krona(request, samp_no):
    """
    Display Krona visualization for taxon abundance of one sample
    """
    Sample = get_sample_model()
    try:
        sample = Sample.objects.get(sample_id=f'samp_{samp_no}')
    except Sample.DoesNotExist:
        raise Http404('no such sample')

    try:
        html = TaxonAbundance.objects.as_krona_html(sample)
    except TaxonAbundance.DoesNotExist:
        raise Http404('no abundance data for sample or error with krona')

    return HttpResponse(html)


class SampleTrackingView(StaffLoginRequiredMixin, SingleTableView):
    template_name = 'omics/sample_tracking.html'
    table_class = SampleTrackingTable
    table_pagination = False

    def get_queryset(self):
        """
        returns a list of dict
        """
        Samples = get_sample_model()
        Dataset = get_dataset_model()
        samples = Samples._meta.base_manager.all().in_bulk()
        self.total_sample_count = len(samples)
        private = dict(
            Dataset._meta.base_manager.all().values_list('pk', 'private')
        )
        tracks = SampleTracking.objects.order_by('sample__pk')
        for i in tracks:
            i.sample = samples[i.sample_id]
        data = []
        for sample, grp in groupby(tracks, lambda x: x.sample):
            row = defaultdict(None)
            row['sample'] = sample
            row['sample_id_num'] = int(sample.sample_id.removeprefix('samp_'))
            row['private'] = private[sample.dataset_id]
            for i in grp:
                row[i.get_flag_display()] = True
            data.append(row)
        return data

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['total_count'] = self.total_sample_count
        ctx['summary_data'] = self.get_summary()
        return ctx

    def get_summary(self):
        data = {
            human_val: 0
            for _, human_val in SampleTracking.Flag.choices
        }
        for row in self.object_list:
            for key in data:
                data[key] += row.get(key, 0)
        return data
