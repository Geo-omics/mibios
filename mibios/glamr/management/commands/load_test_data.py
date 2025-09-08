from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory
import subprocess

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction


class Command(BaseCommand):
    help = ('Load data dump from dump created by "dump_test_data" command.  '
            'Run this on a migrated but otherwise empty database.  The '
            'database should be at the same migration as the one used to make '
            'the dump.  Requires postgresql backend.')

    def add_arguments(self, parser):
        parser.add_argument(
            'dump_file',
            help='A .tar.zst dump file as created by the dump_test_data '
            'command',
        )

    @transaction.atomic()
    def handle(self, *args, **options):
        self.options = options
        self.truncating_done = False

        with TemporaryDirectory() as tmp:
            self.tmp = Path(tmp)
            dump_file = Path(options['dump_file'])
            basename = dump_file.name.removesuffix('.tar.zst')
            tarfile = self.tmp / (basename + '.tar')

            subprocess.run(
                ['unzstd', '-o', str(tarfile), options['dump_file']],
                check=True,
            )

            with TarFile.open(tarfile, 'r') as self.tarf:
                cols_info = None
                for info in iter(self.tarf):
                    # cols files must come right before their repsective dump
                    if info.name.endswith('.cols'):
                        cols_info = info
                    else:
                        self.load(info, cols_info=cols_info)
                        cols_info = None

    NON_EMPTY_TABLES = (
        'django_content_type', 'auth_permission', 'auth_group'
    )
    """ tables that are expected to be non-empty after initial migrations """

    valid_table_names = [
        model._meta.db_table
        for models in apps.all_models.values()
        for model in models.values()
    ]

    def ensure_empty_start(self, cursor):
        """
        Truncate tables that may not be empty after initial migration

        This will only do anything the first time it is called.
        """
        if self.truncating_done:
            return

        for table_name in self.NON_EMPTY_TABLES:
            cursor.execute(f'TRUNCATE {table_name} RESTART IDENTITY CASCADE')
        self.truncating_done = True

    def load(self, tarinfo, cols_info=None):
        """
        Load data from one dump file
        """
        if cols_info is None:
            cols = None
        else:
            if tarinfo.name.removesuffix('.text') != cols_info.name.removesuffix('.cols'):  # noqa:E501
                raise RuntimeError('cols and dump files do not match')
            self.tarf.extract(cols_info, path=self.tmp)
            with open(self.tmp / cols_info.name) as ifile:
                cols = [line.strip() for line in ifile]

        self.tarf.extract(tarinfo, path=self.tmp)
        dump = self.tmp / tarinfo.name
        table_name = dump.name.removesuffix('.dump.text')
        if table_name not in self.valid_table_names:
            raise CommandError(f'not a valid table name: {table_name}')
        print(f'  {table_name:<27}', end=' ', flush=True)
        if cols is None:
            print('      ', end=' ', flush=True)
        else:
            print('[cols]', end=' ', flush=True)

        with connection.cursor() as cur:
            dbname = cur.connection.info.dbname
            if 'dev' not in dbname and 'test' not in dbname:
                raise CommandError(
                    f'\nThe configured DB is {dbname} via {cur.connection.dsn}'
                    f'\nAre you using a test or dev database?'
                )

            if cols is None:
                sql = f'COPY {table_name} FROM STDIN'
            else:
                sql = f'COPY {table_name} ( {", ".join(cols)} ) FROM STDIN'

            with dump.open('r') as dumpf:
                self.ensure_empty_start(cur)
                cur.copy_expert(sql, dumpf)
            count = cur.rowcount

        print(f'{count:>10} [OK]')
