"""
url declarations for the mibios.glamr app
"""
# FIXME: this module is, for now, called url0.  If we were to call it the usual
# url, then the mibios.url will try to include it.  This, when combined with
# setting ROOT_URLCONF to it, as we would do when running as the GLAMR webapp,
# will result in a loop.  Calling the module url0 avoids getting into that loop
# in the first place.  We should revise if the automatic include in mibios.url
# still makes sense.
from functools import partial

from django.conf import settings
from django.urls import include, path
from django.views import defaults

from mibios import urls as mibios_urls
from mibios.omics.views import krona
from . import views


urlpatterns = [
    path('', views.FrontPageView.as_view(), name='frontpage'),
    path('dataset/<int:pk>/samples', views.SampleListView.as_view(), name='dataset_sample_list'),  # noqa: E501
    path('dataset/<int:pk>/', views.DatasetView.as_view(), name='dataset'),
    path('reference/<int:pk>/', views.ReferenceView.as_view(), name='reference'),  # noqa: E501
    path('sample/<int:pk>/', views.SampleView.as_view(), name='sample'),
    path('sample/<int:sample_pk>/krona/<str:stats_field>/', krona, name='krona'),  # noqa:E501
    path('data/<str:model>/', views.TableView.as_view(), name='generic_table'),  # noqa: E501
    path('data/<str:model>/<int:pk>/', views.record_view, name='record'),
    path('data/<str:model>/<int:pk>/overview/', views.OverView.as_view(), name='record_overview'),  # noqa: E501
    path('data/<str:model>/<int:pk>/overview/samples/', views.OverViewSamplesView.as_view(), name='record_overview_samples'),  # noqa: E501
    path('data/<str:model>/<int:pk>/abundance/', views.AbundanceView.as_view(), name='record_abundance'),  # noqa: E501
    path('data/<str:model>/<int:pk>/abundance/<str:sample>/genes/', views.AbundanceGeneView.as_view(), name='record_abundance_genes'),  # noqa: E501
    path('data/<str:model>/<int:pk>/relations/<str:field>/', views.ToManyListView.as_view(), name='relations'),  # noqa: E501
    path('data/<str:model>/<int:pk>/relations/<str:field>/full/', views.ToManyFullListView.as_view(), name='relations_full'),  # noqa: E501
    path('search-adv/', views.SearchView.as_view(), name='search_initial'),
    path('search/<str:model>/', views.ResultListView.as_view(), name='search_result'),  # noqa: E501
    path('filter/<str:model>/', views.FilteredListView.as_view(), name='filter_result'),  # noqa: E501
    path('search-adv/<str:model>/', views.SearchModelView.as_view(), name='search_model'),  # noqa: E501
]

if settings.INTERNAL_DEPLOYMENT:
    urlpatterns.append(path('tables/', include(mibios_urls)))
    urlpatterns.append(path('server-error/', views.test_server_error))


# The default template names are without path, so take up the global name space
# and mibios' templates may take precedence if mibios is listed first in
# INSTALLED_APPS.  So, here we re-declare all the error handler in order to
# pass along our templates with app-specific path.
handler400 = partial(defaults.bad_request, template_name='glamr/errors/400.html')  # noqa: E501
handler404 = partial(defaults.page_not_found, template_name='glamr/errors/404.html')  # noqa: E501
handler500 = partial(defaults.server_error, template_name='glamr/errors/500.html')  # noqa: E501
