from django.urls import path

from . import views
from .admin import admin_site


urlpatterns = [
    path('', views.DatasetListing.as_view(), name='dataset_list'),
    path('dataset/<int:pk>/', views.DatasetDetail.as_view(), name='dataset_detail'),  # noqa:E501
    path('host/<int:pk>/', views.HostDetail.as_view(), name='host_detail'),
    path('sample/<int:pk>/', views.SampleDetail.as_view(), name='sample_detail'),  # noqa:E501
    path('samples/', views.SampleListing.as_view(), name='sample_list'),
    path('tax/', views.TaxBrowser.as_view(), name='tax_browser'),
    path('admin/', admin_site.urls),
]
