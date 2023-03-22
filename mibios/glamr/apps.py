from django.apps import AppConfig as _AppConfig
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
                # raises FieldDoesNotExist if attrs module and Sample model
                # went out of sync:
                field = Sample._meta.get_field(field_name)
                for attr_name, value in attrs.items():
                    setattr(field, attr_name, value)

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
