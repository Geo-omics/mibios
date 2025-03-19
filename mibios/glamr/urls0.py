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

from django.contrib.auth import views as auth_views
from django.http.response import Http404
from django.urls import include, path, re_path
from django.views import defaults
from django.views.decorators.cache import never_cache
from django.views.generic import RedirectView

from mibios import urls as mibios_urls
from mibios.omics import urls as omics_urls
from mibios.omics.views import krona

from . import accounts, views
from .admin import admin_site


kpat = r'(pk:(?P<pk>[\d]+)|(?P<natkey>[\w:-]+))'
""" accession/primary key pattern for RecordView """


def disable_url(*args, **kwargs):
    """
    view for 404 response

    use this as view in urlconf to disable a urls that would otherwise be
    included via some following include()
    """
    raise Http404('disabled')


urlpatterns = [
    path('', views.FrontPageView.as_view(), name='frontpage'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('about/history/', views.AboutHistoryView.as_view(), name='about-history'),  # noqa: E501
    path('contact/', views.ContactView.as_view(), name='contact'),
    path('dataset/<int:set_no>/samples/', views.SampleListView.as_view(), name='dataset_sample_list'),  # noqa: E501
    path('dataset/access/', views.DatasetAccessView.as_view(), name='dataset_access'),  # noqa: E501
    re_path(rf'dataset/{kpat}/$', views.DatasetView.as_view(), name='dataset'),  # noqa: E501
    re_path(rf'reference/{kpat}/$', views.ReferenceView.as_view(), name='reference'),  # noqa: E501
    re_path(rf'sample/{kpat}/$', views.SampleView.as_view(), name='sample'),  # noqa: E501
    path('sample/<int:samp_no>/krona/', krona, name='krona'),
    path('data/', views.AvailableDataView.as_view(), name='available_data'),
    path('data/<str:model>/', views.TableView.as_view(), name='generic_table'),  # noqa: E501
    re_path(rf'data/(?P<model>\w+)/{kpat}/$', views.record_view, name='record'),  # noqa: E501
    path('data/<str:model>/<int:pk>/abundance/', views.AbundanceView.as_view(), name='record_abundance'),  # noqa: E501
    path('data/<str:model>/<int:pk>/abundance/<str:sample>/genes/', views.AbundanceGeneView.as_view(), name='record_abundance_genes'),  # noqa: E501
    path('data/<str:obj_model>/<int:pk>/relations/<str:field>/', views.ToManyListView.as_view(), name='relations'),  # noqa: E501
    path('search/', views.SearchView.as_view(), name='search_initial'),
    path('search/<str:model>/', views.SearchResultListView.as_view(), name='search_result'),  # noqa: E501
    path('filter/adv/<str:model>/', views.AdvFilteredListView.as_view(), name='filter_result'),  # noqa: E501
    path('filter/<str:model>/', views.FilteredListView.as_view(), name='filter_result2'),  # noqa: E501
    path('search-adv/<str:model>/', views.SearchModelView.as_view(), name='search_model'),  # noqa: E501

    # URLs are served depending on settings, e.g. RequiredSettingsMixin
    # see INTERNAL_DEPLOYMENT, ENABLE_TEST_VIEWS, ENABLE_OPEN_ADMIN
    path('dbinfo/', never_cache(views.DBInfoView.as_view()), name='dbinfo'),
    path('admin/login/', RedirectView.as_view(pattern_name='login', query_string=True)),  # noqa:E501
    path('admin/', admin_site.urls),
    # accounts URLs: the mirror most of the contrib.auth.urls
    path('accounts/add/', accounts.AddUserView.as_view(), name='add_user'),
    path('accounts/add/email/<int:user_pk>/', accounts.AddUserEmailView.as_view(), name='add_user_email'),  # noqa:E501
    path('accounts/add/delete/', accounts.DeleteView.as_view(), name='delete_account'),  # noqa:E501
    path('accounts/login/', accounts.LoginView.as_view(), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('accounts/password_change/', accounts.PasswordChangeView.as_view(), name='password_change'),  # noqa:E501
    path('accounts/password_change/done/', accounts.PasswordChangeDoneView.as_view(), name='password_change_done'),  # noqa:E501
    path('accounts/profile/', accounts.UserProfileView.as_view(), name='user_profile'),  # noqa:E501
    path('accounts/reset/<uidb64>/<token>/', accounts.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),  # noqa:E501
    path('accounts/reset/done/', accounts.PasswordResetCompleteView.as_view(), name='password_reset_complete'),  # noqa:E501
    path('errortest/', never_cache(views.test_server_error)),
    path('minitest/', never_cache(views.MiniTestView.as_view())),
    path('basetest/', never_cache(views.BaseTestView.as_view())),
    path('httptest/', never_cache(views.HttpTestView.as_view())),
    path('omics/', include(omics_urls)),
    path('', include(mibios_urls.model_graph_urls))
]


# The default template names are without path, so take up the global name space
# and mibios' templates may take precedence if mibios is listed first in
# INSTALLED_APPS.  So, here we re-declare all the error handler in order to
# pass along our templates with app-specific path.
handler400 = partial(defaults.bad_request, template_name='glamr/errors/400.html')  # noqa: E501
handler404 = partial(defaults.page_not_found, template_name='glamr/errors/404.html')  # noqa: E501
handler500 = partial(defaults.server_error, template_name='glamr/errors/500.html')  # noqa: E501
