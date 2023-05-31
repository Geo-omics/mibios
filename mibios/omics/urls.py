"""
url declarations for the mibios.omics app
"""
from django.urls import path

from . import views


urlpatterns = [
    path('sample-status/', views.SampleStatusView.as_view(), name='sample_status'),  # noqa:E501
]
