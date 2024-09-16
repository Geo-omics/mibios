from django.contrib import admin
from django.contrib import auth
from django.contrib import messages
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse

from .models import AboutInfo, Credit


class AdminSite(admin.AdminSite):
    site_header = 'GLAMR site administration'


class GroupAdmin(auth.admin.GroupAdmin):
    exclude = ['permissions']


class UserAdmin(auth.admin.UserAdmin):
    exclude = ['user_permissions']
    actions = ['setup_account']

    def get_fieldsets(self, request, obj=None):
        # just setting exclude does not work with fieldsets, it seems the form
        # is created fine, but during the rendering process the excluded fields
        # are looked up in the form raising a KeyError.
        #
        # Below is a bit of a hack, it assumes that the data dict only ever
        # contains 'fields' mapping to a list of fields.
        return tuple((
            (label, {k: tuple((i for i in fields if i not in self.exclude))
                     for k, fields in data.items()})
            for label, data
            in super().get_fieldsets(request, obj=obj)
        ))

    @admin.action(description='Account setup/password reset')
    def setup_account(self, request, queryset):
        try:
            user = queryset.get()
        except auth.models.User.DoesNotExist:
            raise Http404('no such user')
        except auth.models.User.MultipleObjectsReturned:
            self.message_user(request, 'Error: you must select a single user'
                              ' for account setup / password reset',
                              messages.ERROR)
            return

        url = reverse('add_user_email', kwargs=dict(user_pk=user.pk))
        return HttpResponseRedirect(url)


admin_site = AdminSite(name='glamr_admin')
admin_site.register(AboutInfo)
admin_site.register(Credit)
admin_site.register(auth.models.User, UserAdmin)
admin_site.register(auth.models.Group, GroupAdmin)
