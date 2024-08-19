"""
WSGI config for mibios project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""
from logging import getLogger
import os
from pathlib import Path

from django.core.wsgi import get_wsgi_application

from mibios import __version__ as version


DEFAULT_SETTINGS = 'mibios.ops.settings'
LOCAL_SETTINGS = 'settings'


log = getLogger(__name__)

if Path(LOCAL_SETTINGS).with_suffix('.py').exists():
    settings = LOCAL_SETTINGS
else:
    settings = DEFAULT_SETTINGS

os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings)
application = get_wsgi_application()
log.info(f'WSGI app loaded settings:{settings} version:{version}')
