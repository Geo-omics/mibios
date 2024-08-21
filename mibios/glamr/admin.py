from django.contrib import admin
from django.contrib.auth.models import Group, User

from .models import AboutInfo, Credit


class AdminSite(admin.AdminSite):
    site_header = 'GLAMR site administration'


admin_site = AdminSite(name='glamr_admin')
admin_site.register(AboutInfo)
admin_site.register(Credit)
admin_site.register(User)
admin_site.register(Group)
