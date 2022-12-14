from django.apps import apps
from django.contrib import admin
from django.db.transaction import atomic
from django.urls import reverse_lazy

from . import get_registry
from .models import ImportFile, Snapshot, ChangeRecord
from .views import HistoryView


app_config = apps.get_app_config('mibios')


class AdminSite(admin.AdminSite):
    site_header = 'Administration'
    index_title = 'Data Curation'
    site_url = reverse_lazy('top')

    def register_all(self):
        """
        Register all the admins

        To be called from AppConfig.ready().  Normally, registration is done
        when this module is imported, but is seems even when following the
        documentation at
        https://docs.djangoproject.com/en/2.2/ref/contrib/admin/#overriding-the-default-admin-site
        we lose all registrations, possibly because site gets re-instantiated
        later, maybe has to do with module auto-discovery.
        """
        for i in get_registry().get_models():
            self.register(i, model_admin_factory(i))

        self.register(ImportFile, ImportFileAdmin)
        self.register(Snapshot, SnapshotAdmin)
        self.register(ChangeRecord, HistoryAdmin)

    def get_app_list(self, request):
        """
        Get the app list but change order a bit

        The auth app shall go last
        """
        auth_admin = None
        app_list = []
        for i in super().get_app_list(request):
            if i['app_label'] == 'auth':
                auth_admin = i
            else:
                app_list.append(i)
        if auth_admin is not None:
            app_list.append(auth_admin)
        return app_list


class ModelAdmin(admin.ModelAdmin):
    exclude = ['history']

    def get_list_display(self, request):
        return self.model.get_fields(with_hidden=True).names

    def save_model(self, request, obj, form, change):
        obj.add_change_record(user=request.user)
        super().save_model(request, obj, form, change)

    def delete_model(self, request, obj):
        obj.add_change_record(user=request.user, is_deleted=True)
        super().delete_model(request, obj)

    @atomic
    def delete_queryset(self, request, queryset):
        for i in queryset:
            i.add_change_record(is_deleted=True, user=request.user)
            # and since Model.delete() won't be called:
            i.change.save()

        super().delete_queryset(request, queryset)

    def history_view(self, request, object_id, extra_context=None):
        record = self.model.objects.get(pk=object_id)
        return HistoryView.as_view()(
                request, record=record, extra_context=extra_context)


def model_admin_factory(model):
    m2m_field_names = [
        i.name for i
        in model.get_fields(with_m2m=True).fields
        if i.many_to_many
    ]
    opts = dict(filter_horizontal=m2m_field_names)
    name = 'Auto' + model._meta.model_name.capitalize() + 'ModelAdmin'
    return type(name, (ModelAdmin,), opts)


class ImportFileAdmin(admin.ModelAdmin):
    actions = None
    list_display = ('timestamp', 'name', 'get_log_url', 'file')


class SnapshotAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'name', 'dbpath', 'jsonpath', 'note')


class HistoryAdmin(admin.ModelAdmin):
    """
    Admin interface for history

    We only want to allow changing comments
    """
    actions = None
    list_display = ('timestamp', 'record_type', 'record_natural',
                    'comment', 'file')
    fields = (
        ('timestamp', 'record_type', 'record_natural', 'record_pk'),
        ('user', 'file', 'line'), ('is_created', 'is_deleted'), 'comment',
    )
    readonly_fields = (
        'timestamp', 'user', 'line', 'record_pk', 'file', 'record_type',
        'record_natural', 'fields', 'is_created', 'is_deleted',
    )

    def has_add_permission(self, request):
        """
        No adding history via the admin interface
        """
        return False

    def has_delete_permission(self, request, obj=None):
        """
        No removing history via the admin interface
        """
        return False
