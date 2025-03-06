import os
from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db.models.fields.files import FieldFile
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.functional import cached_property


class ReadOnlyFieldFile(FieldFile):
    """
    A FieldFile that tries to enforce read-only access.

    Text-mode by default.
    """
    WRITE_MODE_BITS = set(('w', 'x', '+', 'a'))

    def open(self, mode='rt'):
        if self.WRITE_MODE_BITS.isdisjoint(mode):
            return super().open(mode=mode)
        else:
            raise ValueError('mode must be read-only')

    def delete(self, **kwargs):
        raise NotImplementedError('file is read-only')

    def save(self, *args, **kwargs):
        raise NotImplementedError('file is read-only')


class FileOpsMixin:
    """
    Mixin for FileSystemStorage with several file operations
    """
    def move(self, old_name, file):
        """
        Move/rename a file

        old_name: str of old name
        file: a FieldFile, with file.name being the new name
        """
        if file.storage is not self:
            raise ValueError('this storage does not handle this file')
        src = Path(self.location) / old_name
        dst = Path(file.path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        self.prune_empty_dir(src.parent)

    def link_or_copy(self, source, destination):
        """
        If possible, creates a hard link at destination pointing back to
        source.  If both are on different filesystems we fall back to making a
        regular copy.

        source, destination: Instances of Fieldfile
        """
        if destination.storage is not self:
            raise ValueError('this storage does not handle this file')
        link = Path(destination.path)
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.hardlink_to(source.path)
        except OSError as e:
            if 'invalid cross-device link' in str(e).casefold():
                # TODO
                raise NotImplementedError(
                    f'copy not implemented (from {e.__class__.__name__}: {e})'
                ) from e
            raise

    def get_all_files(self):
        """
        Return iterable over all the storages file as registered in the
        database

        The implementing storage must provide this method.
        """
        raise NotImplementedError

    def check_orphans(self):
        """
        Check for any stray files or directories in storage

        Will print unexpected content to stdout.
        """
        location = Path(self.location)
        valid_paths = set((i.path for i in self.get_all_files()))
        for root, dirs, files in os.walk(location):
            root = Path(root)
            if not dirs and not files:
                if root == location:
                    # storage empty ok
                    pass
                else:
                    print(f'empty directory: {root})')
            for i in files:
                file = root / i
                if str(file) not in valid_paths:
                    print(f'orphan: {file}')

    def prune_empty_dir(self, parent):
        """
        Remove empty directories, iterating over their otherwise empty parents
        (below the storage location) and remove those too.

        parent: Instance of Path, the parent directory of a file that was just
        deleted.
        """
        root = Path(self.location)
        while parent.is_relative_to(root):  # guard rail
            if parent == root:
                # never remove the root
                break

            try:
                parent.rmdir()
            except OSError:
                # e.g. is not empty
                break

            parent = parent.parent

    def delete(self, name):
        """
        Replacement for FileSystemStorage.delete() which also pruning empty
        parent directories.
        """
        super().delete(name)
        self.prune_empty_dir(Path(self.path(name)).parent)


class OmicsPipelineStorage(FileSystemStorage):
    """
    Read-only storage where we find analysis pipeline output files

    This storage is used to load data into the DB.  Files here are not
    accessible via the web interface.
    """
    location = settings.OMICS_PIPELINE_DATA
    base_url = None


class LocalPublicStorage(FileOpsMixin, FileSystemStorage):
    """
    Storage for files offered as direct download or otherwise needed by the
    webapp.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('location', settings.LOCAL_STORAGE_ROOT)
        super().__init__(**kwargs)

    @cached_property
    def base_url(self):
        if self._base_url:
            return super().base_url

        PLACEHOLDER = 'PLACEHOLDER'
        try:
            base_url = reverse('file_download', args=(PLACEHOLDER,))
        except NoReverseMatch:
            base_url = None
        else:
            base_url = base_url.removesuffix(PLACEHOLDER + '/')
        return base_url

    def get_all_files(self):
        File = apps.get_model('omics', 'File')
        qs = File.objects.exclude(file_local='')
        return (i.file_local for i in qs)


class GlobusPublicStorage(FileOpsMixin, FileSystemStorage):
    """
    Storage accessible through Globus
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('location', settings.GLOBUS_STORAGE_ROOT)
        kwargs.setdefault('base_url', settings.GLOBUS_DIRECT_URL_BASE)
        super().__init__(**kwargs)

    def get_all_files(self):
        File = apps.get_model('omics', 'File')
        qs = File.objects.exclude(file_globus='')
        return (i.file_globus for i in qs)
