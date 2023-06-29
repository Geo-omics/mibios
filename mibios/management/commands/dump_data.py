from contextlib import contextmanager
from pathlib import Path
import argparse
from subprocess import Popen, PIPE

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connections
from django.db.models import ManyToManyField
from django.db.transaction import atomic

from mibios import get_registry


COMMAND_LOCK = 'DUMP_FILE_UPLOAD_IN_PROGRESS'
DB_SIGNAL = 'REPLICATE_THIS'

MODE_ADD = 'add'
MODE_REPLACE = 'replace'
MODE_UPDATE = 'update'


class DumpBaseCommand(BaseCommand):
    """
    Abstract dump command

    Inheriting classes should implement the do_dump method.
    """
    help = 'dump data in postgres text format'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dumpfiles = []

    def add_arguments(self, parser):
        parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        parser.add_argument(
            '--replicate',
            action='store_true',
            help='Get output dir via settings.REPLICATION_DUMP_PATH (unless '
                 'overridden by -o) and save a transaction manifest file.  '
                 'This is the counterpart to the subscription mode of the '
                 'load_data command.',
        )
        parser.add_argument('-o', '--out-dir', help='output directory.')
        parser.add_argument('--compress', action='store_true',
                            help='compress output with zstd')
        parser.add_argument(
            '-m', '--mode',
            choices=[MODE_ADD, MODE_UPDATE, MODE_REPLACE],
            default=MODE_UPDATE,
            help=f'Insert mode:  {MODE_ADD}: insert rows with non-existing '
            f'IDs, raises an error if existing row ids ar encountered. '
            f'{MODE_REPLACE}: Like "{MODE_ADD}" but truncate the table first. '
            f' {MODE_UPDATE}: Like "{MODE_ADD}" but allow updating existing '
            f'rows.'
        )

    def handle(self, *args, out_dir=None, compress=False, mode=None,
               replicate=False, **options):
        if replicate and out_dir is None:
            out_dir = settings.REPLICATION_DUMP_PATH

        if out_dir is None:
            self.out_dir = Path()
        else:
            self.out_dir = Path(out_dir)
        self.compress = compress
        self.mode = mode

        if replicate:
            with self.replication_sender_mode():
                self.do_dump(**options)
        else:
            self.do_dump(**options)

    @atomic
    def do_dump(self, **options):
        raise NotImplementedError

    @contextmanager
    def replication_sender_mode(self):
        command_lock = self.out_dir / COMMAND_LOCK
        db_signal = self.out_dir / DB_SIGNAL

        try:
            command_lock.mkdir()
        except FileExistsError as e:
            raise CommandError(
                f'An invocation of this command is in progress or a '
                f'previous one did not clean up after itself: '
                f'{e.__class__.__name__} {e}'
            )

        if db_signal.is_file():
            command_lock.rmdir()
            raise CommandError(
                f'A previous dump has not been consumed yet: '
                f'file exists: {db_signal}'
            )

        try:
            yield
            with db_signal.open('w') as ofile:
                for mode, name in self.dumpfiles:
                    ofile.write(f'{mode}\t{name}\n')
        finally:
            command_lock.rmdir()

    def dump_from_queryset(self, cursor, queryset):
        """ dump a single table from given queryset"""
        if not hasattr(cursor, 'copy_expert'):
            raise RuntimeError('requires the postgres DB backend')

        query, params = queryset.query.sql_with_params()
        sql = f'COPY ({query}) TO STDOUT HEADER'
        sql = cursor.mogrify(sql, params)

        opath = self.out_dir / queryset.model._meta.db_table

        if self.compress:
            opath = opath.with_suffix('.tab.zst')
            zstd = ['zstd', '-T6', '-o', str(opath)]
            with Popen(zstd, stdin=PIPE, pipesize=1024 * 1024) as p:
                cursor.copy_expert(sql, p.stdin)
        else:
            opath = opath.with_suffix('.tab')
            with opath.open('w') as ofile:
                cursor.copy_expert(sql, ofile, 1024 * 1024)

        self.dumpfiles.append((self.mode, opath.name))


class Command(DumpBaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            'appnames',
            metavar='appname',
            nargs='+',
            help='Name of one or more apps for which to dump data',
        )
        parser.add_argument(
            '--no-update',
            action='store_true',
            help='Signal to the DB to bail out with error if any row IDs '
                 'exist already, the default is to let the DB load the data '
                 'in UPSERT mode',
        )
        super().add_arguments(parser)

    @atomic
    def do_dump(self, appnames=None, **options):
        for i in appnames:
            self.dump_app_data(i)

    def dump_app_data(self, appname):
        """ dump data for given app """
        models = get_registry().get_models(appname)
        through_models = [
            j.remote_field.through
            for i in models
            for j in i._meta.get_fields()
            if isinstance(j, ManyToManyField)
        ]
        with connections['default'].cursor() as cur:
            for i in models + through_models:
                self.stdout.write(f'{i._meta.model_name} ', ending='')
                self.stdout.flush()
                self.dump_from_queryset(cur, i.objects.all())
                self.stdout.write(f'{cur.rowcount} [OK]\n')
