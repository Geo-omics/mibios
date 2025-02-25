from django.conf import settings
from django.db.models.fields.files import FileField

from mibios.umrad.fields import OldPathField
from .storage import ReadOnlyFieldFile


class DataPathField(OldPathField):
    """
    DEPRECATED use the new PathField instead

    Class remains to support migrations
    """
    description = 'a path under the data root directory'
    default_base = settings.OMICS_PIPELINE_DATA


class ReadOnlyFileField(FileField):
    attr_class = ReadOnlyFieldFile
