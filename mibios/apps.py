from importlib import import_module
import logging

from django import apps
from django.conf import settings
from django.contrib.admin.apps import AdminConfig as UpstreamAdminConfig
from django.utils.module_loading import import_string

from .registry import Registry
from .utils import get_db_connection_info, getLogger, QueryLogFilter


log = getLogger(__name__)


class MibiosConfig(apps.AppConfig):
    name = 'mibios'
    verbose_name = 'Microbiome Data System'

    def ready(self):
        super().ready()

        if settings.DEBUG:
            logging.getLogger('django.db.backends').addFilter(QueryLogFilter())

        # set up registry
        registry = Registry()

        try:
            registry.name = settings.SITE_NAME
        except AttributeError:
            registry.name = self.name
        try:
            registry.verbose_name = settings.SITE_NAME_VERBOSE
        except AttributeError:
            try:
                registry.verbose_name = settings.SITE_NAME
            except AttributeError:
                registry.verbose_name = self.verbose_name

        import_module('mibios')._registry = registry

        # register models here, since django has found them already
        Model = import_string(self.name + '.models.Model')
        for i in apps.apps.get_models():
            if issubclass(i, Model):
                if hasattr(i, 'get_child_info') and i.get_child_info():
                    # model has children, skip
                    continue
                registry.add(i)

                # register all apps with mibios.Models
                app = i._meta.app_label
                if app not in registry.apps:
                    registry.apps[app] = apps.apps.get_app_config(app)

        # register datasets
        for app_conf in registry.apps.values():
            module_name = app_conf.name + '.dataset'
            try:
                registry.add_dataset_module(module_name, app_conf.label)
            except ImportError as e:
                if e.args[0] == f"No module named '{module_name}'":
                    pass
                else:
                    log.debug(f'Failed registering datasets for {app_conf}: '
                              f'{e.__class__.__name__}: {e}')

        # register table view plugins
        for app_conf in registry.apps.values():
            try:
                registry.add_table_view_plugins(app_conf.name + '.views')
            except ImportError:
                pass

        # admin setup below:
        admin = import_string('django.contrib.admin')
        admin.site.register_all()

        # signaling
        import_module('mibios.signals')

        # display some info
        if settings.DEBUG:
            for alias, info in get_db_connection_info().items():
                log.info(f'DB {alias}: {info}')

        info = f'Registry {registry.name} (app:models+datasets):'
        for i in registry.apps.keys():
            info += (
                f' {i}:'
                f'{len(registry.get_models(app=i))}'
                f'+{len(registry.get_datasets(app=i))}'
            )
        log.info(info)


class AdminConfig(UpstreamAdminConfig):
    default_site = 'mibios.admin.AdminSite'
    # name = 'mibios.admin.site'
