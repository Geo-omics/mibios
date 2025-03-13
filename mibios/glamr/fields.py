import decimal
from functools import cached_property

from django.core.validators import URLValidator
from django.db.models import DecimalField, URLField


class OptionalHTTPSURLValidator(URLValidator):
    """ Validates https:// URLs while allowing blanks """
    schemes = ['https']

    def __call__(self, value):
        if value == '':
            # Allow blank
            return
        super().__call__(value)


class OptionalURLField(URLField):
    """ Field for https:// URLs but may remain blank """
    default_validators = [OptionalHTTPSURLValidator()]

    def __init__(self, **kwargs):
        kwargs.setdefault('blank', True)
        super().__init__(**kwargs)


class FreeDecimalField(DecimalField):
    def db_type(self, connection):
        """
        On postgres use the 'numeric' type without parameters

        For other vendors degrade to normal behaviour.
        """
        if connection.vendor == 'postgresql':
            return 'numeric'
        else:
            return super().db_type(connection)

    @cached_property
    def context(self):
        return decimal.Context(
            prec=self.max_digits,
            rounding=decimal.ROUND_HALF_UP,
        )
