from django.contrib import admin

from . import models


class AdminSite(admin.AdminSite):
    pass


admin_site = AdminSite('hamb_admin')
admin_site.register(models.Dataset)
admin_site.register(models.Host)
admin_site.register(models.Sample)
