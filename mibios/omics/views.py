from collections import defaultdict
from itertools import groupby

from django.conf import settings
from django.http import Http404, HttpResponse
from django.views.decorators.cache import patch_cache_control

from django_tables2 import SingleTableView

from mibios.views import StaffLoginRequiredMixin
from . import get_sample_model
from .models import File, SampleTracking, TaxonAbundance
from .tables import FileTable, SampleTrackingTable


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
    # FIXME: exclude_private is defined in/depends on mibios.glamr
    qs = Sample.objects.exclude_private(request.user)
    qs = qs.filter(sample_id=f'samp_{samp_no}')
    try:
        sample = qs.get()
    except Sample.DoesNotExist:
        raise Http404('no such sample')

    try:
        html = TaxonAbundance.objects.as_krona_html(sample)
    except TaxonAbundance.DoesNotExist:
        raise Http404('no abundance data for sample or error with krona')

    resp = HttpResponse(html)
    if request.user.is_authenticated:
        patch_cache_control(resp, private=True)
    return resp


class FileListingView(StaffLoginRequiredMixin, SingleTableView):
    template_name = 'omics/file_listing.html'
    table_class = FileTable
    table_pagination = False

    def get_queryset(self):
        return File.objects.all()


class SampleTrackingView(StaffLoginRequiredMixin, SingleTableView):
    template_name = 'omics/sample_tracking.html'
    table_class = SampleTrackingTable
    table_pagination = False

    def get_queryset(self):
        """
        returns a list of dict
        """
        Samples = get_sample_model()

        sample_fields = (
            'sample_id', 'sample_name', 'analysis_dir', 'read_count',
            'reads_mapped_contigs', 'reads_mapped_genes',
            'biosample',
        )

        samples = (
            Samples._meta.base_manager.all()
            .only(*sample_fields, 'access', 'dataset__dataset_id',)
            .select_related('dataset')
            .in_bulk()
        )

        self.total_sample_count = len(samples)
        tracks = SampleTracking.objects.order_by('sample_id')
        data = []
        for sample_pk, grp in groupby(tracks, lambda x: x.sample_id):
            sample = samples[sample_pk]
            row = defaultdict(None)
            row['sample'] = sample
            row['sample_id_num'] = sample.get_record_id_no()
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
