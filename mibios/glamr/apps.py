from django.apps import AppConfig as _AppConfig
from django.core.checks import register as register_checks
from django.core.exceptions import FieldDoesNotExist
from django.utils.module_loading import import_string

from mibios import __version__


class AppConfig(_AppConfig):
    name = 'mibios.glamr'
    verbose_name = 'GLAMR'
    version = __version__

    def ready(self):
        # Patch externally defined field attributes into the Sample model.
        # This requires the extra attrs module to be prepared.  It seems, that
        # these runtime changes, e.g. overwriting verbose_name attributes, are
        # not detected by the makemigration, which is good.
        try:
            sample_field_attrs = import_string(
                f'{self.name}.extra_field_attributes.extra_sample_field_attrs'
            )
        except ImportError as e:
            print(f'[WARNING] Skip adding extra model field attributes: '
                  f'{e.__class__.__name__}: {e}')
        else:
            Sample = self.get_model('Sample')
            for field_name, attrs in sample_field_attrs.items():
                try:
                    field = Sample._meta.get_field(field_name)
                    for attr_name, value in attrs.items():
                        setattr(field, attr_name, value)
                except FieldDoesNotExist as e:
                    # skip this, just issue a warning
                    print(f'WARNING: Patching field attributes: {e}')
                    print('This means that the Sample model and the '
                          'extra_field_attributes module are out of sync.')

        # Monkey patching concrete fields of Searchable:
        # since the searchvector field is a generated column, we can't let the
        # model save data to it (even if it's just the default). So here we
        # tell the model that it is not concrete.  A hack depending on django
        # internals.  It is sufficient to run Searchable.objects.reindex()
        Searchable = self.get_model('searchable')
        Searchable._meta.concrete_fields = tuple((
            i for i in Searchable._meta.concrete_fields
            if i.name != 'searchvector'
        ))

        # register sample jobs here to avoid cyclic import issues
        job_registry = import_string('mibios.omics.tracking.registry')
        job_registry.register_from_module('mibios.glamr.jobs')

        # register checks
        register_checks()(import_string('mibios.glamr.accounts.accounts_check'))  # noqa:E501
