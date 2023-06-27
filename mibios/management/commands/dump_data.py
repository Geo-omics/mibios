from pathlib import Path
from subprocess import Popen, PIPE

from django.core.management.base import BaseCommand
from django.db import connections
from django.db.models import ManyToManyField
from django.db.transaction import atomic

from mibios import get_registry


class Command(BaseCommand):
    help = 'dump data in postgres text format'

    def add_arguments(self, parser):
        parser.add_argument(
            'appnames',
            metavar='appname',
            nargs='+',
            help='Name of one or more apps for which to dump data',
        )
        parser.add_argument('-o', '--out-dir', help='output directory')
        parser.add_argument('--compress', action='store_true',
                            help='compress output with zstd')

    def handle(self, *args, appnames=None, out_dir=None, compress=False,
               **options):
        if out_dir is None:
            self.out_dir = Path()
        else:
            self.out_dir = Path(out_dir)
        self.compress = compress

        with atomic():
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
                self.dump_table(cur, i)
                self.stdout.write('[OK]\n')

    def dump_table(self, cursor, model):
        """ dump a single table """
        query, params = model.objects.all().query.sql_with_params()
        if params != ():
            raise RuntimeError(
                'unexpectedly got non-empty params with {model}: {params}'
            )

        sql = f'COPY ({query}) TO STDOUT HEADER'
        opath = self.out_dir / model._meta.db_table

        if self.compress:
            opath = opath.with_suffix('.tab.zst')
            zstd = ['zstd', '-T6', '-o', str(opath)]
            with Popen(zstd, stdin=PIPE, pipesize=1024 * 1024) as p:
                return cursor.copy_expert(sql, p.stdin)
        else:
            opath = opath.with_suffix('.tab')
            with opath.open('w') as ofile:
                return cursor.copy_expert(sql, ofile, 1024 * 1024)
