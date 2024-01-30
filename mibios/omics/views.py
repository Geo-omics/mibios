from django.conf import settings
from django.http import Http404, HttpResponse

from django_tables2 import SingleTableView

from . import get_sample_model
from .models import TaxonAbundance
from .tables import SampleStatusTable


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


class SampleStatusView(RequiredSettingsMixin, SingleTableView):
    required_settings = 'INTERNAL_DEPLOYMENT'
    template_name = 'omics/sample_status.html'
    table_class = SampleStatusTable
    model = get_sample_model()
    table_pagination = False

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['total_count'], ctx['summary_data'] = self.get_summary()
        return ctx

    def get_summary(self):
        data = dict()
        for i in SampleStatusTable._meta.fields:
            field = self.model._meta.get_field(i)
            if field.get_internal_type() == 'BooleanField':
                data[i] = 0

        total = 0
        for obj in self.model.objects.only(*data.keys()):
            total += 1
            for flag in data.keys():
                if getattr(obj, flag):
                    data[flag] += 1

        return total, data
