from datetime import date
from logging import getLogger
from pathlib import Path

from django.apps import AppConfig as _AppConfig
from django.conf import settings


from mibios import __version__


class AppConfig(_AppConfig):
    name = 'mibios.omics'
    verbose_name = 'omics'
    version = __version__

    def ready(self):
        log_dir = getattr(settings, 'IMPORT_DIFF_DIR', None)
        if log_dir:
            # find a unique log file name and set up the log handler
            path = Path(log_dir) / 'foo'
            log = getLogger('omics_sample_loader')
            stem, _, suffix = \
                Path(log.handlers[0].baseFilename).name.partition('.')
            today = date.today()
            name = f'{stem}.{today}.{suffix}'
            num = 0
            while path.with_name(name).exists():
                num += 1
                name = f'{stem}.{today}.{num}.{suffix}'

            log.handlers[0].baseFilename = path.with_name(name)
