from django.conf import settings
from django.core.files.storage import FileSystemStorage


class ReadOnlyFileSystemStorage(FileSystemStorage):
    """
    Treat storage as read-only
    """
    def delete(self, name):
        raise NotImplementedError('storage is read-only')

    def save(self, *args, **kwargs):
        raise NotImplementedError('storage is read-only')

    def open(self, name, mode='rb'):
        breakpoint()
        if 'w' in mode or 'x' in mode or '+' in mode or 'a' in mode:
            raise NotImplementedError('storage is read-only')
        return super().open(name, mode=mode)


class OmicsPipelineStorage(ReadOnlyFileSystemStorage):
    """
    Read-only storage where we find analysis pipeline output files

    This storage is used to load data into the DB.  Files here are not
    accessible via the web interface.
    """
    location = settings.OMICS_PIPELINE_DATA
    base_url = None


class LocalPublicStorage(FileSystemStorage):
    """
    Storage for files offered as direct download or otherwise needed by the
    webapp.
    """
    location = settings.FILESTORAGE_ROOT
    base_url = settings.FILESTORAGE_URL


class GlobusPublicStorage(FileSystemStorage):
    """
    Storage accessible through Globus
    """
    location = settings.GLOBUS_STORAGE_ROOT
    base_url = settings.GLOBUS_DIRECT_URL_BASE
