from django.conf import settings

from mibios.umrad.fields import OldPathField


class DataPathField(OldPathField):
    """
    DEPRECATED use the new PathField instead

    Class remains to support migrations
    """
    description = 'a path under the data root directory'
    default_base = settings.OMICS_DATA_ROOT
