from pathlib import Path
from shutil import copy2

from django.apps import apps
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db.models.fields.files import FieldFile
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.functional import cached_property

from .utils import check_modtime_microseconds


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

    def link_or_copy(self, source, destination, exist_ok=True):
        """
        If possible, creates a hard link at destination pointing back to
        source.  If both are on different filesystems we fall back to making a
        regular copy.

        source, destination:
            Instances of FieldFile
        exists_ok [bool]:
            If True then and the destination file exists and is of the right
            size then return with no further action.  If the size mismatches
            then raise a RuntimeError.  If False and the destination exists,
            then a FileExistsError is raised.
        """
        if destination.storage is not self:
            raise ValueError('this storage does not handle this file')
        # destination.path will fails if destination file doesn't exist yet
        dst = Path(self.location) / destination.name
        if not dst.is_relative_to(self.location):
            raise ValueError('destination is not under storage location')
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.hardlink_to(source.path)
        except FileExistsError as e:
            if exist_ok:
                try:
                    dest_size = dst.stat().st_size
                except FileNotFoundError:
                    # broken symlink
                    raise RuntimeError(f'destination is broken symlink: {dst}') from e
                if dest_size == source.size:
                    return
                else:
                    raise RuntimeError(f'file exists but size differs: {dst}') from e
            raise
        except OSError as e:
            if 'invalid cross-device link' in str(e).casefold():
                # Note that copy2 would happily overwrite existing destinations
                # but hardlink_to will catch those before we'll ever get here.
                copy2(source.path, dst)
            else:
                raise

    def get_all_files(self):
        """
        Return iterable over all the storages file as registered in the
        database

        The implementing storage must provide this method.
        """
        raise NotImplementedError

    def find(self):
        """
        Find all files and directories in storage
        """
        location = Path(self.location)
        for curdir, dirs, files in location.walk():
            reldir = curdir.relative_to(location)
            if not dirs and not files:
                if curdir == location:
                    # storage empty ok
                    return
                else:
                    yield ('empty', reldir)

            yield from (('dir', reldir / i) for i in dirs)
            yield from (('file', reldir / i) for i in files)

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

    @cached_property
    def supports_microseconds(self):
        """
        In certain setups (devel or testing) storage might be accessed via
        sshfs which may not support sub-second file modification times via
        stat() causing the modtime comparisions with omics.models.File to fail.

        This methods will try to detect this situation and return False to
        indicate the lack of precision and maybe allow leniency.  Will return
        True in all other cases including certain errors.
        """
        return check_modtime_microseconds(self.location)


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

        FAKE_PATH = 'fake_path'
        # Get the base url from our url conf.  There are two considerations:
        # (1) reverse() requires a path argument, which must be removed again.
        # (2) To make the url, our parent class will run urllib.parse.urljoin()
        # on the base url and the file path.  This would remove the last part
        # of the base url's path unless it ends in a slash.  So, this means the
        # urlconf must give us a path ending in a slash followed by the passed
        # placeholder path.  If not then the storage's urls will be a mess, but
        # no checks are done here.
        try:
            base_url = reverse('file_download', args=(FAKE_PATH,))
        except NoReverseMatch:
            base_url = None
        else:
            base_url = base_url.removesuffix(FAKE_PATH)
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
