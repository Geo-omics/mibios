from django.conf import settings
from django.contrib import admin
from django.contrib.auth.models import User

from .models import AboutInfo, Credit


class FakeUser(User):
    def has_module_perms(self, *args, **kwargs):
        return True

    def has_perm(self, *args, **kwargs):
        return True


class OpenAdminSite(admin.AdminSite):
    site_header = 'GLAMR site administration'

    def has_permission(self, request):
        try:
            # try for a staff user, who must be in the DB
            user = User.objects.filter(is_staff=True).first()
        except User.DoesNotExist:
            # the fake user is good to look at the admin site but when trying
            # to save changes will fail on the log entry creation as that needs
            # a user in the database.
            user = None

        request.user = user or FakeUser()
        return True


if settings.INTERNAL_DEPLOYMENT and settings.ENABLE_OPEN_ADMIN:
    admin_site = OpenAdminSite(name='glamr_admin')
    admin_site.register(AboutInfo)
    admin_site.register(Credit)
else:
    # allow for import statement in e.g. urls.py
    admin_site = None
