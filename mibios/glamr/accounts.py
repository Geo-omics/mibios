from django.contrib.auth.models import Group, Permission, User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import views as auth_views
from django.core import checks
from django.db import Error as DB_Error
from django.views.generic import DetailView

from .models import Dataset


class LoginView(auth_views.LoginView):
    template_name = 'accounts/login.html'


class PasswordChangeView(auth_views.PasswordChangeView):
    template_name = 'accounts/password_change.html'


class UserProfileView(LoginRequiredMixin, DetailView):
    model = User
    template_name = 'accounts/profile.html'

    def get_object(self, queryset=None):
        return self.request.user

    def get_my_datasets(self):
        """
        Get restricted dataset the current user can access
        """
        return Dataset.objects.filter(restricted_to__user=self.request.user)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['datasets'] = self.get_my_datasets()
        return ctx


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


def accounts_check(app_configs, **kwargs):
    """
    Run some checks on user and group accounts

    * checks if the glamr-staff group exists
    * checks if glamr-staff has correct permissions
    * checks if all and only is_staff users are member of the glamr-staff group

    To be run as part of the system checks.
    """
    name = GLAMR_STAFF_GROUP
    errors = []

    try:
        if not User.objects.exists() and not Group.objects.exists():
            # be quiet if there are no users/groups at all
            # e.g. for non-production
            return []
    except DB_Error:
        # be quiet here, there are clearly some other issues going on
        # e.g. tables don't exist
        return []

    try:
        staff_grp = Group.objects.get(name='glamr-staff')
    except Group.DoesNotExist:
        errors.append(
            checks.Warning(
                f'The {name} group does not exist.',
                hint='Create glamr-staff group.',
                id='glamr.W001',
            )
        )
        # further checks don't make sense
        return errors

    perms = set(staff_grp.permissions.values_list('codename', flat=True))
    if perms:
        if missing := GLAMR_STAFF_PERMISSIONS - perms:
            errors.append(
                checks.Warning(
                    f'The {name} group is missing some permissions: '
                    f'{sorted(missing)}',
                    hint='Add permissions via admin (check mibios.glamr.admin '
                    'module if permissions are not excluded from interface)',
                    id='glamr.W002',
                )
            )
        if extra := perms - GLAMR_STAFF_PERMISSIONS:
            errors.append(
                checks.Warning(
                    f'The {name} group has extra permissions: '
                    f'{sorted(extra)}',
                    hint='Remove extra permissions via admin '
                    '(check mibios.glamr.admin module if permissions are not '
                    'excluded from interface)',
                    id='glamr.W003',
                )
            )
    else:
        errors.append(
            checks.Warning(
                f'The {name} group has no permissions at all.',
                hint='Add permissions (see GLAMR_STAFF_PERMISSIONS in '
                'mibios.glamr.accounts) via admin or run '
                'create_staff_group() also in the accounts module.',
                id='glamr.W004',
            )
        )

    if missing := User.objects.filter(is_staff=True).exclude(groups=staff_grp):
        missing = ', '.join((i.username for i in missing))
        errors.append(
            checks.Error(
                f'Staff users are not member of the {staff_grp.name} group',
                hint=f'Add users: {missing} to the {staff_grp.name} group or'
                ' set is_staff to False',
                id='glamr.E005',
            )
        )

    if bad := staff_grp.user_set.filter(is_staff=False):
        errors.append(
            checks.Error(
                f'Some non-staff users are member of the {staff_grp.name} '
                f'group',
                hint=f'Remove users: {bad} to the {staff_grp.name} group or '
                'set is_staff to True',
                id='glamr.E006',
            )
        )
    return errors
