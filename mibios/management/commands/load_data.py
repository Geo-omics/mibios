from pathlib import Path

from psycopg2.sql import SQL, Identifier

from django.core.management.base import BaseCommand, CommandError
from django.db import connections
from django.db.transaction import atomic

from .dump_data import MODE_ADD, MODE_REPLACE, MODE_UPDATE


class DryRunRollback(Exception):
    pass


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
            modes = []
            data_files = []
            with job_listing.open() as ifile:
                for line in ifile:
                    mode, name = line.split()
                    modes.append(mode)
                    infile = job_listing.parent / name
                    if not infile.is_file():
                        raise CommandError('no such file: {infile}')
                    data_files.append(infile)
        else:
            modes = [mode] * len(file_args)
            data_files = [Path(i) for i in file_args]

        self.conn = connections['default']
        try:
            with atomic():
                if MODE_REPLACE in modes:
                    self.do_truncate(*(
                        f for m, f
                        in zip(modes, data_files)
                        if m == MODE_REPLACE)
                    )

                for mode, infile in zip(modes, data_files):
                    self.stdout.write(f'{infile} ...', ending='')
                    self.stdout.flush()
                    if mode == MODE_UPDATE:
                        total, changed = self.update_from_file(infile)
                        self.stdout.write(f'  {changed}/{total} [OK]\n')
                    elif mode == MODE_ADD or mode == MODE_REPLACE:
                        rowcount = self.add_from_file(infile)
                        self.stdout.write(f'  {rowcount} [OK]\n')
                    else:
                        raise ValueError('bad mode')
                if dry_run:
                    raise DryRunRollback
        except DryRunRollback:
            self.stderr.write('dry run rollback')
        else:
            self.stdout.write('Cleanup: deleting input files... ', ending='')
            for i in data_files:
                i.unlink()
            job_listing.unlink()
            self.stdout.write('[Done]')

    def add_from_file(self, path):
        """ load data into one table from given path """
        table_name, _, _ = Path(path).name.partition('.')

        with open(path) as ifile:
            head = ifile.readline().split()
        columns = SQL(',').join((Identifier(i) for i in head))

        sql = SQL('COPY {table_name} ({columns}) FROM STDIN WITH HEADER')
        sql = sql.format(table_name=Identifier(table_name), columns=columns)
        with open(path) as ifile, self.conn.cursor() as cur:
            cur.copy_expert(sql, ifile)
            return cur.rowcount

    def update_from_file(self, path):
        """ update a table from given path """
        table_name, _, _ = path.name.partition('.')
        tmp_table = 'tmp_' + table_name
        with path.open() as ifile:
            head = ifile.readline().split()
        columns = SQL(', ').join((Identifier(i) for i in head))
        excl_cols = SQL(', ').join((
            SQL('.').join([Identifier('excluded'), Identifier(i)])
            for i in head
        ))

        table_name = Identifier(table_name)
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

        with open(path) as ifile, self.conn.cursor() as cur:
            cur.execute(sql0)
            cur.copy_expert(sql1, ifile)
            total_rows = cur.rowcount

        with self.conn.cursor() as cur:
            cur.execute(sql2)
            new_or_changed_rows = cur.rowcount
            cur.execute(sql3)

        return total_rows, new_or_changed_rows

    def do_truncate(self, *infiles):
        table_names = [Path(i).name.partition('.')[0] for i in infiles]
        tables = SQL(', ').join((Identifier(i) for i in table_names))
        sql = SQL('TRUNCATE {tables} CASCADE')
        sql = sql.format(tables=tables)

        table_names = ', '.join(table_names)
        self.stdout.write(
            f'Erasing all data in: {table_names} + cascade', ending=''
        )
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.stdout.write(' [OK]')
