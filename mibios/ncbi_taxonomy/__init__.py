"""
The NCBI Taxonomy Database
"""
from pathlib import Path
from subprocess import run
import sys

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from mibios import get_registry


REMOTE_HOST = 'ftp.ncbi.nih.gov'
REMOTE_PATH = '/pub/taxonomy/new_taxdump'
ARCHIVE_NAME = 'new_taxdump.tar.gz'
README = 'taxdump_readme.txt'

WGET = '/usr/bin/wget'
MD5SUM = '/usr/bin/md5sum'
TAR = '/bin/tar'


DUMP_FILES = {
    'citation': 'citations.dmp',
    'deletednode': 'delnodes.dmp',
    'division': 'division.dmp',
    'gencode': 'gencode.dmp',
    'host': 'host.dmp',
    'mergednodes': 'merged.dmp',
    'taxname': 'names.dmp',
    'taxnode': 'nodes.dmp',
    'typematerial': 'typematerial.dmp',
    'typematerialtype': 'typeoftype.dmp',
}
""" map model names to dump files """


def get_data_source_path():
    """
    Return path to data download/source directory
    """
    try:
        return Path(settings.NCBI_TAXONOMY_DUMP_DIR)
    except AttributeError:
        return Path.cwd()


def download_latest(dest=None):
    """
    Download latest version of the taxonomy to destination directory
    """
    if dest is None:
        try:
            dest = settings.NCBI_TAXONOMY_DUMP_DIR
        except AttributeError as e:
            raise ImproperlyConfigured(f'{e.name} setting not configured')

    dest = Path(dest)
    if not dest.is_dir():
        dest.mkdir()
        print(f'Created directory: {dest}', file=sys.stderr)

    url = f'https://{REMOTE_HOST}:{REMOTE_PATH}/{README}'
    run([WGET, '--backups=1', url], cwd=dest, check=True)
    url = f'https://{REMOTE_HOST}:{REMOTE_PATH}/{ARCHIVE_NAME}'
    run([WGET, '--backups=1', url], cwd=dest, check=True)
    run([WGET, '--backups=1', url + '.md5'], cwd=dest, check=True)
    run([MD5SUM, '--check', ARCHIVE_NAME + '.md5'], cwd=dest, check=True)
    run([TAR, 'vxf', ARCHIVE_NAME], cwd=dest, check=True)


"""
Model names in order of FK-relational dependencies.  Models earlier in the
order must be populated before later ones so that FK fields can be set (as the
PKs must be known).  The data can be deleted in reverse order.
"""
REL_DEPEND_ORDER = [
    'gencode',
    'division',
    'deletednode',
    'taxnode',
    'taxname',
    'mergednodes',
    # 'host',
    # 'typematerialtype',
    # 'typematerial',
    'citation',
]


def load(dry_run=False):
    """
    Load all data from the downloaded source

    Not loading until we know we need these:
        taxidlineage.dmp
        rankedlineage.dmp
        fullnamelineage.dmp
    """
    models = get_registry().apps['ncbi_taxonomy'].models
    for i in REL_DEPEND_ORDER:
        model = models[i]
        print(f'Loading NCBI Taxonomy "{model._meta.verbose_name}":')
        model.loader.load(dry_run=dry_run)


def erase():
    from mibios.umrad.model_utils import delete_all_objects_quickly
    for i in get_registry().apps['ncbi_taxonomy'].get_models():
        delete_all_objects_quickly(i)
