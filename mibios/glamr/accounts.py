from django import forms
from django.contrib import messages
from django.contrib.auth import views as auth_views
from django.contrib.auth.forms import AuthenticationForm, UsernameField
from django.contrib.auth.models import Group, Permission, User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core import checks
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import Error as DB_Error
from django.db.transaction import atomic
from django.db.utils import DatabaseError
from django.http.response import Http404
from django.template.loader import render_to_string
from django.urls import reverse, reverse_lazy
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.views.generic import DetailView, FormView, TemplateView


from mibios.views import StaffLoginRequiredMixin
from .models import Dataset
from .views import BaseMixin


STAFF_GROUP_NAME = 'glamr-staff'
""" name of the glamr staff group """


class AddUserForm(forms.Form):
    group = forms.ModelChoiceField(
        queryset=Group.objects.all(),
        empty_label=None,
    )
    emails = forms.CharField(widget=forms.Textarea)

    def clean_emails(self):
        data = self.cleaned_data['emails']
        data = data.split()
        if not data:
            raise ValidationError('input field is empty')

        errors = []
        for item in data:
            try:
                validate_email(item)
            except ValidationError as e:
                errors.append(e)
        if errors:
            errors.insert(0, 'The email field must list valid email addresses')
            raise ValidationError(errors)

        return data


class AuthenticationForm2(AuthenticationForm):
    username = UsernameField(
        widget=forms.TextInput(attrs={'autofocus': True}),
        label='Your email address',
    )

    def clean_username(self):
        data = self.cleaned_data.get('username')
        if data.endswith(('@umich.edu', '@med.umich.edu')):
            data, _, _ = data.rpartition('@')
        return data


class DeleteAccountForm(forms.Form):
    username = forms.CharField(widget=forms.HiddenInput)


class AddUserView(StaffLoginRequiredMixin, BaseMixin, FormView):
    template_name = 'accounts/add_user.html'
    form_class = AddUserForm

    def form_valid(self, form):
        try:
            users, new_count = self.create_users(**form.cleaned_data)
        except DatabaseError as e:
            form.add_error(
                None,
                f'Failed storing new user records in the DB: '
                f'{e.__class__.__name__}: {e}',
            )
            return self.form_invalid(form)

        ctx = self.get_context_data(
            group=form.cleaned_data['group'],
            users=users,
            new_count=new_count,
            some_last_login=any((i.last_login for i in users))
        )
        return self.render_to_response(ctx)

    @atomic
    def create_users(self, group=None, emails=None):
        """ Creates the users as needed and adds them to the group """
        users = []
        new_count = 0
        for i in emails:
            if i.endswith(('@umich.edu', '@med.umich.edu')):
                # keep uniqenames for local folks
                username, _, _ = i.partition('@')
            else:
                username = i
            user, new = User.objects.get_or_create(
                email=i,
                defaults={
                    'username': username,
                    'is_staff': group.name == STAFF_GROUP_NAME,
                },
            )
            if new:
                new_count += 1
            users.append(user)

        group.user_set.add(*users)
        return users, new_count


class AddUserEmailView(StaffLoginRequiredMixin, BaseMixin, TemplateView):
    template_name = 'accounts/add_user_email.html'
    subject_template_name = 'accounts/welcome_email_subject.html'
    email_template_name = 'accounts/welcome_email.html'

    def get_email(self):
        """
        generate the welcome email

        This borrows a lot from PasswordResetForm from the auth app.
        """
        try:
            user = User.objects.get(pk=self.kwargs['user_pk'])
        except User.DoesNotExist:
            raise Http404('no such user')

        if not user.email:
            raise Http404('user has no email adress')

        domain = get_current_site(self.request)
        subject = render_to_string(
            self.subject_template_name,
            dict(domain=domain),
        )
        protocol = 'https' if self.request.is_secure() else 'http'
        path = reverse("password_reset_confirm", kwargs=dict(
            uidb64=urlsafe_base64_encode(force_bytes(user.pk)),
            token=default_token_generator.make_token(user),
        ))
        reset_url = f'{protocol}://{domain}{path}'
        body = render_to_string(
            self.email_template_name,
            {
                'domain': domain,
                'user': user,
                'reset_url': reset_url,
                'datasets': Dataset.objects.filter(restricted_to__user=user)
            }
        )
        return user.email, subject, body, reset_url

    def get_context_data(self, **ctx):
        ctx['address'], ctx['subject'], ctx['body'], ctx['reset_url'] \
            = self.get_email()
        return ctx


class DeleteView(LoginRequiredMixin, FormView):
    template_name = 'accounts/delete.html'
    success_url = reverse_lazy('logout')
    form_class = DeleteAccountForm

    def get_initial(self):
        ini = super().get_initial()
        ini['username'] = self.request.user.username
        return ini

    def form_valid(self, form):
        user = self.request.user
        if form.cleaned_data['username'] != user.username:
            raise Http404('bad account delete form submission')

        if user.is_staff:
            messages.add_message(
                self.request,
                messages.ERROR,
                'Staff user accounts can not be deleted this way.',
            )
            self.success_url = reverse_lazy('user_profile')

        else:
            user.is_active = False
            try:
                user.save()
            except DatabaseError:
                messages.add_message(
                    self.request,
                    messages.ERROR,
                    'Account deletion failed.  You may try again.',
                )
                self.success_url = reverse_lazy('user_profile')
            else:
                messages.add_message(
                    self.request,
                    messages.SUCCESS,
                    f'The account for {user.username} is deleted',
                )
        return super().form_valid(form)


class LoginView(auth_views.LoginView):
    template_name = 'accounts/login.html'
    form_class = AuthenticationForm2


class PasswordChangeView(auth_views.PasswordChangeView):
    template_name = 'accounts/password_change.html'


class PasswordResetConfirmView(auth_views.PasswordResetConfirmView):
    template_name = 'accounts/password_reset_confirm.html'


class PasswordResetCompleteView(auth_views.PasswordResetCompleteView):
    template_name = 'accounts/password_reset_complete.html'


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


STAFF_PERMISSIONS = set([
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


def create_staff_group():
    """
    convenience function to set up the staff group

    But users need to be added manually.
    """
    grp, _ = Group.objects.get_or_create(name=STAFF_GROUP_NAME)
    perms = Permission.objects.filter(codename__in=STAFF_PERMISSIONS)
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
    name = STAFF_GROUP_NAME
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
        if missing := STAFF_PERMISSIONS - perms:
            errors.append(
                checks.Warning(
                    f'The {name} group is missing some permissions: '
                    f'{sorted(missing)}',
                    hint='Add permissions via admin (check mibios.glamr.admin '
                    'module if permissions are not excluded from interface)',
                    id='glamr.W002',
                )
            )
        if extra := perms - STAFF_PERMISSIONS:
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
                hint='Add permissions (see STAFF_PERMISSIONS in '
                'mibios.glamr.accounts) via admin or run '
                'create_staff_group() also in the accounts module.',
                id='glamr.W004',
            )
        )

    if missing := User.objects.filter(is_staff=True).exclude(groups=staff_grp):
        missing = ', '.join((i.username for i in missing))
        errors.append(
            checks.Warning(
                f'Staff users are not member of the {staff_grp.name} group',
                hint=f'Add users: {missing} to the {staff_grp.name} group or'
                ' set is_staff to False',
                id='glamr.W005',
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
