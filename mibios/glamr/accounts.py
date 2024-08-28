from django.contrib.auth.models import Group, Permission, User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import views as auth_views
from django.views.generic import DetailView


class LoginView(auth_views.LoginView):
    template_name = 'accounts/login.html'


class PasswordChangeView(auth_views.PasswordChangeView):
    template_name = 'accounts/password_change.html'


class UserProfileView(LoginRequiredMixin, DetailView):
    model = User
    template_name = 'accounts/profile.html'

    def get_object(self, queryset=None):
        return self.request.user


class PasswordChangeDoneView(UserProfileView):
    extra_context = dict(password_change_done=True)


GLAMR_STAFF_PERMISSIONS = set([
    'add_logentry',
    'change_logentry',
    'delete_logentry',
    'view_logentry',
    'add_group',
    'change_group',
    'delete_group',
    'view_group',
    'add_user',
    'change_user',
    'delete_user',
    'view_user',
    'add_aboutinfo',
    'change_aboutinfo',
    'delete_aboutinfo',
    'view_aboutinfo',
    'add_credit',
    'change_credit',
    'delete_credit',
    'view_credit',
])
""" permissions the staff group should have """

GLAMR_STAFF_GROUP = 'glamr-staff'
""" name of the glamr staff group """


def create_staff_group():
    """
    convenience function to set up the staff group

    But users need to be added manually.
    """
    grp, _ = Group.objects.get_or_create(name=GLAMR_STAFF_GROUP)
    perms = Permission.objects.filter(codename__in=GLAMR_STAFF_PERMISSIONS)
    grp.permissions.set(perms)
    return grp
