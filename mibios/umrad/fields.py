from functools import cached_property
from pathlib import Path

from django.core.exceptions import ValidationError
from django.db.models import Field, CharField


def path_exists_validator(value):
    if not value.exists():
        raise ValidationError(f'path does not exist: {value}')


class PathPrefixValidator:
    """ prefix validator for the PathField """
    def __init__(self, prefix):
        if prefix is None:
            self.parts = None
        else:
            # assume pathlib.Path
            self.parts = prefix.parts

    def __call__(self, value):
        if self.parts is None:
            raise ValidationError(
                'The path value can not be validated, it looks like the '
                'PathField.path attribute was never set'
            )
        val_parts = value.parts
        for i, a in enumerate(self.parts):
            try:
                b = val_parts[i]
            except IndexError:
                raise ValidationError('prefix does not match: too short')

            if a != b:
                raise ValidationError(
                    f'prefix does not match, part {i + 1} is "{b}", '
                    f'expected "{a}"'
                )

    def deconstruct(self):
        """
        Deconstructor to support serialization for migrations
        """
        if self.parts is None:
            prefix = None
        else:
            prefix = Path().joinpath(*self.parts)
        return ('mibios.umrad.fields.PathPrefixValidator', (prefix,), {})

    def __eq__(self, other):
        # __eq__ is here to support serialization for migrations
        return self.parts == other.parts


class PrefixValidator:
    prefix = ''

    def __init__(self, prefix):
        if prefix is not None:
            self.prefix = prefix

    def __call__(self, value):
        return value.startswith(self.prefix)


class AccessionField(CharField):
    description = 'dataset-specific accession'
    prefix = None
    DEFAULT_LENGTH = 32

    def __init__(self, verbose_name=None, name=None, prefix=None, **kwargs):
        # We have a default max length and also by default unique=True and no
        # prefix, name.  Verbose name will be generated from the model.
        # TODO: what about generating the name, too?
        kwargs.setdefault('max_length', self.DEFAULT_LENGTH)
        kwargs.setdefault('unique', True)
        super().__init__(verbose_name, name, **kwargs)

        if prefix is not None:
            self.prefix = prefix
            self.validators.append(PrefixValidator(prefix))

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if kwargs.get("max_length") == self.DEFAULT_LENGTH:
            del kwargs['max_length']
        if self.unique:
            del kwargs['unique']
        else:
            kwargs['unique'] = False
        if self.prefix is not None:
            kwargs['prefix'] = self.prefix
        return name, path, args, kwargs

    def contribute_to_class(self, cls, name, **kwargs):
        set_verbose_name = self.verbose_name is None
        super().contribute_to_class(cls, name, **kwargs)
        if set_verbose_name:
            # verbose_name should now have been set via
            # set_attributes_from_name()
            # FIXME: line below causes this default to be used in a model
            # derived from an abstract model that declares an AccessionField:
            # ??? self.verbose_name = cls._meta.verbose_name + ' id'
            pass

    def from_db_value(self, value, expression, connection):
        if self.prefix is not None and value is not None:
            return self.prefix + value
        return value

    def get_prep_value(self, value):
        if self.prefix is not None and value is not None:
            if value.startswith(self.prefix):
                value = value[len(self.prefix):]
        return value


class PathField(Field):
    """
    Like FilePathField but value is of type pathlib.Path if not None

    This field stores path values relative to a common root.  The root may be
    None to support certain scenarios where such root is not configured.  Only
    the relative path below the common root is stored on the database and the
    values reconstituted when passed back to the django app.
    """
    DEFAULT_MAX_LENGTH = 200

    def __init__(self, root=None, **kwargs):
        self._root_kwarg = root
        if root is None:
            self.root = None
        elif callable(root):
            self.root = root()
        else:
            self.root = Path(root)
        kwargs.setdefault('max_length', self.DEFAULT_MAX_LENGTH)
        super().__init__(**kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['root'] = self._root_kwarg
        if self.max_length == self.DEFAULT_MAX_LENGTH:
            del kwargs['max_length']
        return name, path, args, kwargs

    def get_internal_type(self):
        return 'FilePathField'

    @cached_property
    def root(self):
        if self._root_kwarg is None:
            return None
        elif callable(self._root_kwarg):
            return self._root_kwarg()
        else:
            return Path(self._root_kwarg)

    def _put_under_root(self, value):
        try:
            # empty str becomes ./ here
            value = Path(value)
        except Exception as e:
            raise ValidationError(str(e)) from e

        if self.root is None:
            return value

        if value.is_relative_to(self.root):
            return value

        if value.is_absolute():
            raise ValidationError('absolute path must be relative to root')

        return self.root / value

    def from_db_value(self, value, expression, connection):
        if value is None:
            return None
        else:
            return self._put_under_root(value)

    def to_python(self, value):
        if value is None:
            return None
        else:
            return self._put_under_root(value)

    def get_prep_value(self, value):
        if value is None:
            return value
        elif self.root is None:
            return str(value)
        else:
            return str(value.relative_to(self.root))

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))


class OldPathField(Field):
    description = 'a file-system-path-like field'
    default_base = './'  # str

    def __init__(self, *args, base=None, exists=True,
                 null=True, default=None, **kwargs):
        super().__init__(*args, null=null, default=default, **kwargs)
        if base is None:
            self.base = Path(self.default_base)
        else:
            self.base = Path(self.default_base) / base

        if exists:
            self.default_validators = [path_exists_validator]

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['base'] = str(self.base.relative_to(self.default_base))
        return name, path, args, kwargs

    def get_internal_type(self):
        return 'TextField'

    def from_db_value(self, value, expression, connection):
        if value is None:
            return None

        return self.base / value

    def to_python(self, value):
        if isinstance(value, Path):
            return value
        if value is None:
            return None
        # TODO: verify that this can never fail
        return Path(value)

    def get_prep_value(self, value):
        return str(value.relative_to(self.base))

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))
