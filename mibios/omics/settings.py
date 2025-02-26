"""
settings for the mibios.omics app
"""
from os import environ
from pathlib import Path

from mibios.ops.settings import *  # noqa:F403

# hook up the apps
INSTALLED_APPS.append('mibios.ncbi_taxonomy.apps.AppConfig')  # noqa:F405
INSTALLED_APPS.append('mibios.umrad.apps.AppConfig')  # noqa:F405
INSTALLED_APPS.append('mibios.omics.apps.AppConfig')  # noqa:F405

# defaults for those swappable models (they are strings <appname>.<model_name>)
OMICS_SAMPLE_MODEL = 'omics.Sample'
OMICS_DATASET_MODEL = 'omics.Dataset'

# register logging
LOGGING['loggers']['omics'] = LOGGING['loggers']['mibios']  # noqa:F405

OMICS_LOADING_LOG = None

# path to sample block list (or leave empty)
# The block list should list, one per line, the sample_id of samples that
# should not be processed beyond meta data.  Empty lines or line starting with
# a # are ignored.
SAMPLE_BLOCKLIST = ''


def get_db_settings(db_dir='.', db_infix=''):
    """
    Call this to set DATABASE

    db_dir:  Directory where to store the DB, without trailing slash
    db_infix: optional infix to distinguish alternative DB
    """
    db_file = f'omics{db_infix}.sqlite3'
    db_mode = 'ro' if environ.get('MIBIOS_DB_RO') else 'rwc'

    return {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': f'file:{db_dir}/{db_file}?mode={db_mode}',
            'OPTIONS': {'uri': True},
        },
    }


DATABASES = get_db_settings()

# Set to True to expose extra, non-public functionality via the web-frontend,
# leave it at False for public-facing deployments
INTERNAL_DEPLOYMENT = False

OMICS_PIPELINE_ROOT = Path('/nosuchdirectory')
OMICS_PIPELINE_DATA = OMICS_PIPELINE_ROOT / 'data' / 'omics'
GLOBUS_STORAGE_ROOT = Path('/nosuchdirectory')
""" path to root of publicly accessible directory tree on staging server, leave
at None on other deployments """

KRONA_CACHE_DIR = './krona-cache'

GLOBUS_DIRECT_URL_BASE = None
""" root Globus url for publicly accessible direct download links """

GLOBUS_FILE_APP_URL_BASE = None
""" root Globus url for file app, publicly shared directory """

LOCAL_STORAGE_ROOT = None
""" path to local filestorage for webapp use """

FILESTORAGE_URL = None
""" base URL for files; cf. MEDIA_URL """

OMICS_CHECKOUT_FILE = None
""" path to the file checkout listing """

STORAGES = {
    'staticfiles': {
        'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage',
    },
    'omics_pipeline': {
        'BACKEND': 'mibios.omics.storage.OmicsPipelineStorage',
    },
    'local_public': {
        'BACKEND': 'mibios.omics.storage.LocalPublicStorage',
    },
    'globus_public': {
        'BACKEND': 'mibios.omics.storage.GlobusPublicStorage',
    },
}
