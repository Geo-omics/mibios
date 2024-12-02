from datetime import datetime
from pathlib import Path
import shutil
import tempfile

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Save backup of user data if data has changed since earlier backup.'

    BACKUP_BASE = 'user_backup_'
    BACKUP_SUFFIX = '.json'

    def add_arguments(self, parser):
        parser.add_argument(
            '--outdir',
            default=None,
            help='Directory to which backups get written.'
        )

    def handle(self, *args, outdir=None, **options):
        if outdir is None:
            outdir = Path()
        else:
            outdir = Path(outdir)

        if not outdir.is_dir():
            raise CommandError(f'no such directory: {outdir}')

        # get most recent backup
        old = None
        for i in outdir.glob(f'{self.BACKUP_BASE}*{self.BACKUP_SUFFIX}'):
            if old is None or old.name < i.name:
                old = i

        today = datetime.now().date()  # prints as YYYY-MM-DD
        fname = self.BACKUP_BASE + str(today) + self.BACKUP_SUFFIX
        with tempfile.TemporaryDirectory() as tmpd:
            new = Path(tmpd) / fname
            call_command(
                'dumpdata',
                'auth.User',
                format='json',
                indent=4,
                database='default',
                natural_foreign=True,
                natural_primary=True,
                output=new,
            )

            if old and old.read_text() == new.read_text():
                self.stdout.write(f'No changes, {old} is already up-to-date\n')
                return

            dst = outdir / new.name
            shutil.copyfile(new, dst)
            self.stdout.write(f'Backup written to {dst}\n')
            if old and old.name == new.name:
                # multiple backups for same day
                self.stdout.write('[WARNING] File with same name existed and '
                                  'got overwritten!')
