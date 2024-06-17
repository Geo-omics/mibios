"""
url declarations for the mibios.omics app
"""
from django.urls import path

from . import views


urlpatterns = [
    path('sample-tracker/', views.SampleTrackingView.as_view(), name='sample_tracking'),  # noqa:E501
]
