from django.urls import path

from . import views
from .admin import admin_site


urlpatterns = [
    path('', views.DatasetListing.as_view(), name='dataset_list'),
    path('abundance/', views.ASVAbundanceListing.as_view(), name='asvabundance_list'),  # noqa:E501
    path('asv/<int:asvnum>/', views.ASVDetail.as_view(), name='asv_detail'),  # noqa:E501
    path('asv/<int:asvnum>/abundance/', views.SingleASVAbundList.as_view(), name='asv_abund_list'),  # noqa:E501
    path('dataset/<int:pk>/', views.DatasetDetail.as_view(), name='dataset_detail'),  # noqa:E501
    path('host/<int:pk>/', views.HostDetail.as_view(), name='host_detail'),
    path('host/<int:pk>/samples', views.HostSampleListing.as_view(), name='host_sample_list'),  # noqa:E501
    path('sample/<int:pk>/', views.SampleDetail.as_view(), name='sample_detail'),  # noqa:E501
    path('sample/<int:pk>/abundance/', views.SampleAbundList.as_view(), name='sample_abund_list'),  # noqa:E501
    path('samples/', views.SampleListing.as_view(), name='sample_list'),
    path('tax/', views.TaxBrowser.as_view(), name='tax_browser_root'),
    path('tax/<int:taxid>/', views.TaxBrowser.as_view(), name='tax_browser'),
    path('taxon/<int:taxid>/', views.TaxonDetail.as_view(), name='taxon_detail'),  # noqa:E501
    path('admin/', admin_site.urls),
]
