from django.contrib import admin
from django.contrib import auth
from django.contrib import messages
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse

from .accounts import STAFF_GROUP_NAME
from .models import AboutInfo, Credit


class AdminSite(admin.AdminSite):
    site_header = 'GLAMR site administration'


class UserChangeForm(auth.forms.UserChangeForm):
    def clean_groups(self):
        """ automatically set the staff group membership based on is_staff """
        # this should be a Group queryset
        groups = self.cleaned_data.get('groups')
        group_names = None
        is_staff = self.cleaned_data.get('is_staff')
        if STAFF_GROUP_NAME in ((i.name for i in groups)):
            if not is_staff:
                # remove group
                group_names = [i.name for i in groups
                               if i.name != STAFF_GROUP_NAME]
        else:
            if is_staff:
                # add user to staff group
                group_names = [i.name for i in groups] + [STAFF_GROUP_NAME]

        if group_names is None:
            # keep as-is
            return groups
        else:
            self.cleaned_data['groups'] = \
                auth.models.Group.objects.filter(name__in=group_names)
            return self.cleaned_data['groups']


class GroupAdmin(auth.admin.GroupAdmin):
    exclude = ['permissions']


class UserAdmin(auth.admin.UserAdmin):
    form = UserChangeForm
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
