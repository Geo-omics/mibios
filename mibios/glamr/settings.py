"""
settings for the GLAMR app
"""
from django.contrib.messages import constants as message_constants

from mibios.omics.settings import *  # noqa:F403

LOGGING['loggers']['mibios.glamr'] = LOGGING['loggers']['mibios']  # noqa:F405

# add our app
INSTALLED_APPS.append('mibios.glamr.apps.AppConfig')  # noqa:F405

TEST_RUNNER = 'mibios.glamr.tests.DiscoverRunner'

# add django-filters
INSTALLED_APPS.append('django_filters')  # noqa:F405

# add crispy forms
INSTALLED_APPS.append('crispy_forms')  # noqa:F405
CRISPY_ALLOWED_TEMPLATE_PACKS = 'bootstrap4'
CRISPY_TEMPLATE_PACK = 'bootstrap4'

# find bootstrap icons as template
TEMPLATES[0]['DIRS'].append('/usr/share/bootstrap-icons/svg/')  # noqa:F405
BASE_TEMPLATE_NAME = 'glamr/base.html'

# override mibios' urls since glamr has it's own
ROOT_URLCONF = 'mibios.glamr.urls0'

# swappable models (these are strings "<app_name>.<model_name>")
OMICS_SAMPLE_MODEL = 'glamr.Sample'
OMICS_DATASET_MODEL = 'glamr.Dataset'

DJANGO_TABLES2_TEMPLATE = 'django_tables2/bootstrap4.html'

# path to so file without .so suffix, e.g. './spellfix'
# Setting this enables search suggestions
SPELLFIX_EXT_PATH = None

# use messaging with bootstrap 5 CSS classes :
MESSAGE_TAGS = {
    message_constants.DEBUG: 'alert-primary',
    message_constants.INFO: 'alert-info',
    message_constants.SUCCESS: 'alert-success',
    message_constants.WARNING: 'alert-warning',
    message_constants.ERROR: 'alert-danger',
}

MEDIA_ROOT = None
MEDIA_URL = None

LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'frontpage'
LOGOUT_REDIRECT_URL = 'frontpage'

# base URL for 'file_download'
FILE_DOWNLOAD_URL = '/download/'

# Set to True to enable URLs for testing
ENABLE_TEST_VIEWS = False

# Set to True to activate an unauthenticated admin interface, only to be used
# in restricted environments please
ENABLE_OPEN_ADMIN = False

# To enable file download via mod_xsendfile, set this to correspond to the
# httpd's XSenfFilePath directive.
HTTPD_FILESTORAGE_ROOT = None
