from django.urls import path

from . import views
from .admin import admin_site


urlpatterns = [
    path('', views.FrontPageView.as_view(), name='top'),
    path('admin/', admin_site.urls),
]
