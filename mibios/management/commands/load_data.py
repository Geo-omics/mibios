from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from psycopg2.sql import SQL, Identifier

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import connections
from django.db.models import Model
from django.db.transaction import atomic

from .dump_data import MODE_ADD, MODE_REPLACE, MODE_UPDATE


class DryRunRollback(Exception):
    pass


@dataclass
class DumpFile:
    path: Path
    mode: str = None
    db_table: str = None
    model: Model = None
    slice_num: int = None

    _table2model = {
        i._meta.db_table: i
        for i in apps.get_models(include_auto_created=True)
    }

    def __post_init__(self):
        name_parts = self.path.name.split('.')
        if len(name_parts) == 3:
            slice_num = name_parts[1]
        else:
            slice_num = 1

        if self.db_table is None:
            self.db_table = name_parts[0]
            if not self.db_table:
                raise ValueError('db_table part is empty string')

        if self.model is None:
            try:
                self.model = self._table2model[self.db_table]
            except KeyError:
                raise LookupError(f'no model with db table "{self.db_table}"')

        if self.slice_num is None:
            try:
                self.slice_num = int(slice_num)
            except ValueError as e:
                raise ValueError('failed parsing filename') from e


class Command(BaseCommand):
    help = 'load data from postgres text format'

    def add_arguments(self, parser):
        parser.add_argument(
            'file_args',
            metavar='infiles',
            nargs='+',
            help='One or more input files in postgres dump text format.  '
                 'With --replicate this is the replication transaction '
                 'listing',
        )
        parser.add_argument(
            '-n', '--dry-run',
            action='store_true',
            help='Dry run, do not actually commit anything to the database.'
        )
        parser.add_argument(
            '--replicate',
            action='store_true',
            help='Load data deposited by dump_data --replicate'
        )
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

    def handle(self, *args, file_args=None, mode=None, dry_run=False,
               replicate=False, **options):

        if replicate:
            if len(file_args) > 1:
                raise CommandError('In replication mode only a single input '
                                   'file is accepted.')
            job_listing = Path(file_args[0])
            dumps = []
            with job_listing.open() as ifile:
                for line in ifile:
                    mode, name = line.split()
                    dump = DumpFile(job_listing.parent / name)
                    if not dump.path.is_file():
                        raise CommandError('no such file: {dump.path}')
                    dump.mode = mode
                    dumps.append(dump)
        else:
            dumps = [DumpFile(path=Path(i), mode=mode) for i in file_args]

        self.conn = connections['default']

        try:
            with atomic():
                if MODE_REPLACE in [i.mode for i in dumps]:
                    self.do_truncate(
                        *(i for i in dumps if i.mode == MODE_REPLACE)
                    )

                with self.conn.cursor() as cur:
                    cur.execute('SET session_replication_role = replica')

                for i in dumps:
                    self.stdout.write(f'{i.path.name} ...', ending='')
                    self.stdout.flush()
                    t0 = monotonic()
                    if i.mode == MODE_UPDATE:
                        total, changed = self.update_from_file(i)
                        result_info = f'{changed}/{total}'
                    elif i.mode == MODE_ADD or i.mode == MODE_REPLACE:
                        total = self.add_from_file(i)
                        result_info = f'{total}'
                    else:
                        raise ValueError('bad mode')
                    t1 = monotonic()
                    rate = f' ({total / (t1 - t0):.0f}/s)' if total else ''
                    self.stdout.write(f'  {result_info}{rate} [OK]\n')
                if dry_run:
                    raise DryRunRollback
        except DryRunRollback:
            self.stderr.write('dry run rollback')
        else:
            self.stderr.write('Cleanup: deleting input files... ', ending='')
            for i in dumps:
                i.path.unlink()
            job_listing.unlink()
            self.stderr.write('[Done]')

    def add_from_file(self, dump):
        """ load data into one table from given dump file """
        with open(dump.path) as ifile:
            head = ifile.readline().split()
        columns = SQL(',').join((Identifier(i) for i in head))

        sql = SQL('COPY {table_name} ({columns}) FROM STDIN WITH HEADER')
        sql = sql.format(table_name=Identifier(dump.db_table), columns=columns)
        with dump.path.open() as ifile, self.conn.cursor() as cur:
            cur.execute('show session_replication_role')
            cur.copy_expert(sql, ifile)
            return cur.rowcount

    def update_from_file(self, dump):
        """ update a table from given dump file """
        tmp_table = 'tmp_' + dump.db_table
        with dump.path.open() as ifile:
            head = ifile.readline().split()
        columns = SQL(', ').join((Identifier(i) for i in head))
        excl_cols = SQL(', ').join((
            SQL('.').join([Identifier('excluded'), Identifier(i)])
            for i in head
        ))

        table_name = Identifier(dump.db_table)
        tmp_table = Identifier(tmp_table)

        sql0 = SQL('CREATE TEMPORARY TABLE {tmp_table} (LIKE {table_name})')
        sql1 = SQL('COPY {tmp_table} ({columns}) FROM STDIN WITH HEADER')
        sql2 = SQL('INSERT INTO {table_name} as t OVERRIDING SYSTEM VALUE '
                   'SELECT * FROM {tmp_table} '
                   'ON CONFLICT (id) DO UPDATE '
                   'SET ({columns}) = ROW ({excl_cols}) '
                   'WHERE (t.*) IS DISTINCT FROM (excluded.*)')
        sql3 = SQL('DROP TABLE {tmp_table}')

        sql0 = sql0.format(tmp_table=tmp_table, table_name=table_name)
        sql1 = sql1.format(tmp_table=tmp_table, columns=columns)
        sql2 = sql2.format(table_name=table_name, tmp_table=tmp_table,
                           columns=columns, excl_cols=excl_cols)
        sql3 = sql3.format(tmp_table=tmp_table)

        with dump.path.open() as ifile, self.conn.cursor() as cur:
            cur.execute(sql0)
            cur.copy_expert(sql1, ifile)
            total_rows = cur.rowcount

        with self.conn.cursor() as cur:
            cur.execute(sql2)
            new_or_changed_rows = cur.rowcount
            cur.execute(sql3)

        return total_rows, new_or_changed_rows

    def do_truncate(self, *dumps):
        tables = SQL(', ').join((Identifier(i.db_table) for i in dumps))
        sql = SQL('TRUNCATE {tables} CASCADE')
        sql = sql.format(tables=tables)

        table_names = ', '.join((i.db_table for i in dumps))
        self.stdout.write(
            f'Erasing all data in: {table_names} + cascade', ending=''
        )
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.stdout.write(' [OK]')