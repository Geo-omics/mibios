from pathlib import Path

from psycopg2.sql import SQL, Identifier

from django.core.management.base import BaseCommand
from django.db import connections
from django.db.transaction import atomic, set_rollback


class Command(BaseCommand):
    help = 'load data from postgres text format'

    def add_arguments(self, parser):
        parser.add_argument(
            'infiles',
            metavar='dumpfile',
            nargs='+',
            help='One or more input files in postgres dump text format',
        )
        parser.add_argument(
            '-n', '--dry-run',
            action='store_true',
            help='Dry run, do not actually commit anything to the database.'
        )
        parser.add_argument(
            '-u', '--update',
            action='store_true',
            help='Allow updating existing rows.  By default, rows can only be '
                 'added and trying to add an existing row raise an '
                 'psycopg2.errors.UniqueViolation.',
        )

    def handle(self, *args, infiles=None, update=False, dry_run=False,
               **options):
        with atomic():
            for i in infiles:
                self.stdout.write(f'{i} ...', ending='')
                self.stdout.flush()
                if update:
                    total, changed = self.update_from_file(i)
                    self.stdout.write(f'  {changed}/{total} [OK]\n')
                else:
                    rowcount = self.load_from_file(i)
                    self.stdout.write(f'  {rowcount} [OK]\n')
            if dry_run:
                print('ROLLBACK')
                set_rollback(True)

    def load_from_file(self, path):
        """ load data into one table from given path """
        table_name, _, _ = Path(path).name.partition('.')
        with open(path) as ifile:
            head = ifile.readline().split()
        columns = SQL(',').join((Identifier(i) for i in head))

        sql = SQL('COPY {table_name} ({columns}) FROM STDIN WITH HEADER')
        sql = sql.format(table_name=Identifier(table_name), columns=columns)
        with open(path) as ifile, connections['default'].cursor() as cur:
            cur.copy_expert(sql, ifile)
            return cur.rowcount

    def update_from_file(self, path):
        """ update a table from given path """
        table_name, _, _ = Path(path).name.partition('.')
        tmp_table = 'tmp_' + table_name
        with open(path) as ifile:
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

        with open(path) as ifile, connections['default'].cursor() as cur:
            cur.execute(sql0)
            cur.copy_expert(sql1, ifile)
            total_rows = cur.rowcount
            cur.execute(sql2)
            new_or_changed_rows = cur.rowcount
            cur.execute(sql3)
        return total_rows, new_or_changed_rows
