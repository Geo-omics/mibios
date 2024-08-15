"""
WSGI config for mibios project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""
from datetime import datetime
import os
from pathlib import Path
import sys

from django.core.wsgi import get_wsgi_application

from mibios import __version__ as version


DEFAULT_SETTINGS = 'mibios.ops.settings'
LOCAL_SETTINGS = 'settings'

if Path(LOCAL_SETTINGS).with_suffix('.py').exists():
    settings = LOCAL_SETTINGS
else:
    settings = DEFAULT_SETTINGS

os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings)
application = get_wsgi_application()
print(f'[{datetime.now()}] wsgi settings:{settings} version:{version}',
      file=sys.stderr)
