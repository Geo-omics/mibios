from contextlib import contextmanager
from io import SEEK_END
from pathlib import Path
import argparse
from subprocess import Popen, PIPE
from time import monotonic

from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connections
from django.db.transaction import atomic

from mibios.utils import sort_models_by_fk_relations
from . import WriteMixin

COMMAND_LOCK = 'DUMP_FILE_UPLOAD_IN_PROGRESS'
DB_SIGNAL = 'REPLICATE_THIS'

MODE_ADD = 'add'
MODE_REPLACE = 'replace'
MODE_UPDATE = 'update'

MODE_OVERRIDES = {
        ('contenttypes', 'contenttype'): MODE_REPLACE,
        ('glamr', 'uniqueword'): MODE_REPLACE,
        ('glamr', 'searchable'): MODE_REPLACE,
}


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
                self.do_dump(dep_sort=True, **options)
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

    def dump_from_queryset(self, cursor, queryset, out_path=None):
        """ dump a single table from given queryset"""
        if not hasattr(cursor, 'copy_expert'):
            raise RuntimeError('requires the postgres DB backend')

        query, params = queryset.query.sql_with_params()
        sql = f'COPY ({query}) TO STDOUT HEADER'
        sql = cursor.mogrify(sql, params)
        model = queryset.model

        if out_path is None:
            opath = self.out_dir / model._meta.db_table
        else:
            opath = Path(out_path)

        if self.compress:
            opath = opath.with_suffix('.tab.zst')
            zstd = ['zstd', '-T6', '-o', str(opath)]
            with Popen(zstd, stdin=PIPE, pipesize=1024 * 1024) as p:
                cursor.copy_expert(sql, p.stdin)
        else:
            opath = opath.with_suffix('.tab')
            with opath.open('w') as ofile:
                cursor.copy_expert(sql, ofile, 1024 * 1024)

        mode = MODE_OVERRIDES.get(
            (model._meta.app_label, model._meta.model_name),
            self.mode
        )
        self.dumpfiles.append((mode, opath.name))


class Command(WriteMixin, DumpBaseCommand):

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
        parser.add_argument(
            '--slice-size',
            type=int,
            default=None,
            help='Partition dump files to have at most this many rows.'
        )
        super().add_arguments(parser)

    @atomic
    def do_dump(self, appnames=None, slice_size=None, dep_sort=False,
                **options):
        if slice_size is not None and slice_size < 1:
            raise CommandError('slice_size must be a positive integer')

        if len(set(appnames)) < len(appnames):
            raise CommandError('Did you give the same app multiple times?')

        # get all models
        models = []
        for i in appnames:
            conf = apps.get_app_config(i)
            for j in conf.get_models(include_auto_created=True):
                models.append(j)

        if dep_sort:
            models = sort_models_by_fk_relations(models)

        # do the dumping
        with connections['default'].cursor() as cur:
            for i in models:
                self.dump_model(cur, i, slice_size)

    def dump_model(self, cursor, model, slice_size):
        if slice_size:
            for i in model._meta.get_fields():
                if i.many_to_one and i.related_model is model:
                    self.warn(
                        f'Overriding slicing of {model._meta.model_name}!'
                    )
                    slice_size = None
                    break

        lastpk = -1
        slice_num = 1
        while True:

            if slice_size is None:
                infix = ''
                qs = model.objects.all()
            else:
                infix = '.' + str(slice_num)
                qs = model.objects.filter(pk__gt=lastpk).order_by('pk')
                qs = qs[:slice_size]

            opath = self.out_dir / f'{model._meta.db_table}{infix}.tab'

            self.info(f'{model._meta.model_name} ', end='')
            t0 = monotonic()
            self.dump_from_queryset(cursor, qs, opath)
            t1 = monotonic()
            count = cursor.rowcount
            rate = f' ({count / (t1 - t0):.0f}/s)' if count else ''
            self.info(f'{count} rows{rate} ', end='')

            # Empty tables are written out as a single file with no
            # rows (maybe just a header), whether or not we do slices.
            # If we do slices and the total row count happened to be a
            # multiple of the slice size, then after all rows are
            # dumped, we don't know we're done yet, but the next dump
            # will have zero row count and will be deleted.
            if slice_num > 1 and cursor.rowcount == 0:
                opath.unlink()
                self.notice('[no further slice]')
            else:
                self.success('[OK]')

            if slice_size is None or cursor.rowcount < slice_size:
                break

            slice_num += 1
            lastpk = self.get_last_row_id(opath)

    def get_last_row_id(self, path):
        """
        Get the last row id in given dump file

        Assumes the row id (of type integer) is the first <tab> separated
        colum.  It expects there to be a header row.
        """
        # This method implements an equivalent to "tail -n1 | cut -f1" in a
        # shell.  It works by reading a small amount of data at the end of the
        # file and trying to divide that into two or more lines, in which case
        # we've got the last line to work with.  If not, then we go back but
        # double the amount of data read.  In the worst case we don't find any
        # newlines but will read the whole file three times over (twice during
        # the doubling phase, and if, for the last doubling, we start reading
        # at position 1, there will be another final read of almost the entire
        # file from position 0.)
        INITIAL_READ_SIZE = 1000
        with path.open() as ifile:
            end = ifile.seek(0, SEEK_END)
            pos = max(0, end - INITIAL_READ_SIZE)
            while True:
                ifile.seek(pos)

                lineno = None
                line = None
                for lineno, line in enumerate(ifile):
                    pass

                if lineno is None:
                    # empty file
                    raise RuntimeError(f'empty file: {path}')

                if lineno == 0:
                    if pos == 0:
                        # file has only a single line, assuming this is the
                        # header and there is no data row
                        raise RuntimeError(
                            f'file has only a single line: {path}'
                        )

                    # didn't get complete last line, doubling seek distance
                    # from end, but don't go past 0
                    pos = max(0, pos - (end - pos))
                    continue

                rowid = line.split()[0]

                try:
                    return int(rowid)
                except ValueError:
                    raise RuntimeError(
                        f'failed parsing rowid: {path=} {pos=} {lineno=} '
                        f'{rowid=} {line=}'
                    )
