from django.urls import path

from . import views
from .models import Abundance


app_name = 'mibios_seq'

_abundance = Abundance._meta.model_name

urlpatterns = [
    path(
        _abundance + '/export-shared-form/',
        views.ExportSharedFormView.as_view(data_name=_abundance),
        name='export_shared_form',
    ),
    path(
        _abundance + '/export-shared/',
        views.ExportSharedView.as_view(data_name=_abundance),
        name='export_shared',
    ),
]
