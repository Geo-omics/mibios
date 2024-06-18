from django.apps import AppConfig as _AppConfig
from django.utils.module_loading import import_string


from mibios import __version__


class AppConfig(_AppConfig):
    name = 'mibios.omics'
    verbose_name = 'omics'
    version = __version__

    def ready(self):

        step_reg = import_string('mibios.omics.tracking.registry')
        step_reg.register_from_module('mibios.omics.steps')
