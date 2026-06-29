"""
url declarations for the mibios.omics app
"""
from django.urls import path

from . import views


urlpatterns = [
    path('sample-tracker/', views.SampleTrackingView.as_view(), name='sample_tracking'),  # noqa:E501
    path('dataset-tracker/', views.DatasetTrackingView.as_view(), name='dataset_tracking'),  # noqa:E501
    path('download-files/', views.FileListingView.as_view(), name='file_listing'),  # noqa:E501
    path('contig/<int:pk>/sequence/', views.ContigSequenceView.as_view(), name='contig_seq'),  # noqa:E501
    path('import-timeline/', views.ImportTimelineView.as_view(), name='import_timeline'),  # noqa:E501
]
