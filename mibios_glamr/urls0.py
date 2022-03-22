"""
url declarations for the mibios_glamr app
"""
# FIXME: this module is, for now, called url0.  If we were to call it the usual
# url, then the mibios.url will try to include it.  This, when combined with
# setting ROOT_URLCONF to it, as we would do when running as the GLAMR webapp,
# will result in a loop.  Calling the module url0 avoids getting into that loop
# in the first place.  We should revise if the automatic include in mibios.url
# still makes sense.
from django.urls import include, path

from mibios import urls as mibios_urls
from . import views


urlpatterns = [
    path('', views.DemoFrontPageView.as_view(), name='frontpage'),
    path('model-graphs/', views.ModelGraphView.as_view(), name='graphs'),
    path('tables/', include(mibios_urls)),
    path('dataset/<int:pk>/samples', views.SampleListView.as_view(), name='dataset_sample_list'),  # noqa: E501
    path('dataset/<int:pk>/', views.DatasetView.as_view(), name='dataset'),
    path('reference/<int:pk>/', views.ReferenceView.as_view(), name='reference'),  # noqa: E501
    path('sample/<int:pk>/', views.SampleView.as_view(), name='sample'),
    path('data/<str:model>/<int:pk>/', views.RecordView.as_view(), name='record'),  # noqa: E501
    path('data/<str:model>/<int:pk>/overview/', views.OverView.as_view(), name='record_overview'),  # noqa: E501
    path('data/<str:model>/<int:pk>/overview/samples/', views.OverViewSamplesView.as_view(), name='record_overview_samples'),  # noqa: E501
    path('data/<str:model>/<int:pk>/abundance/', views.AbundanceView.as_view(), name='record_abundance'),  # noqa: E501
    path('data/<str:model>/<int:pk>/relations/<str:field>/', views.ToManyListView.as_view(), name='relations'),  # noqa: E501
    path('data/<str:model>/<int:pk>/relations/<str:field>/full/', views.ToManyFullListView.as_view(), name='relations_full'),  # noqa: E501
    path('search/hits/', views.SearchHitView.as_view(), name='search_hits'),  # noqa: E501
]
