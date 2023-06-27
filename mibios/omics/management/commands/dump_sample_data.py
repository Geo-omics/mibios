from pathlib import Path
from subprocess import Popen, PIPE

from psycopg2.sql import SQL, Literal

from django.core.management.base import BaseCommand, CommandError
from django.db import connections
from django.db.transaction import atomic

from mibios.omics import get_sample_model
from mibios.omics.models import Alignment, Contig, Gene


class Command(BaseCommand):
    help = 'dump omics data for given sample(s)'

    def add_arguments(self, parser):
        parser.add_argument(
            'sample_ids',
            metavar='SAMPLE',
            nargs='+',
            help='One or more sample_id values',
        )
        parser.add_argument('-o', '--out-dir', help='output directory')
        parser.add_argument('--compress', action='store_true',
                            help='compress output with zstd')

    def handle(self, *args, out_dir=None, compress=False, sample_ids=None,
               **options):
        if out_dir is None:
            out_dir = Path()
        else:
            out_dir = Path(out_dir)

        Sample = get_sample_model()
        samples = Sample.objects.filter(sample_id__in=sample_ids)

        if len(samples) != len(sample_ids):
            raise CommandError('some samples not found')

        Contig_taxa = Contig._meta.get_field('taxa').remote_field.through
        Gene_taxa = Gene._meta.get_field('taxa').remote_field.through

        qsets = [
            Contig.objects.filter(sample__in=samples),
            Contig_taxa.objects.filter(contig__sample__in=samples),
            Gene.objects.filter(sample__in=samples),
            Gene_taxa.objects.filter(gene__sample__in=samples),
            Alignment.objects.filter(gene__sample__in=samples),
        ]

        with atomic(), connections['default'].cursor() as cur:
            for i in qsets:
                self.stdout.write(f'{i.model._meta.model_name} ', ending='')
                self.dump_table(cur, i, out_dir, compress=compress)
                self.stdout.write('[OK]\n')

    def dump_table(self, cursor, queryset, out_dir, compress=True):
        """
        Dump a single table from given queryset
        """
        if not hasattr(cursor, 'copy_expert'):
            raise RuntimeError('requires the postgres DB backend')

        query_sql, params = queryset.query.sql_with_params()
        sql = f'COPY ({query_sql}) TO STDOUT HEADER'
        sql = SQL(sql.replace('%s', '{}'))  # FIXME: not quite kosher
        sql = sql.format(*(Literal(i) for i in params))
        opath = out_dir / queryset.model._meta.db_table
        if compress:
            opath = opath.with_suffix('.tab.zst')
            zstd = ['zstd', '-T6', '-o', str(opath)]
            with Popen(zstd, stdin=PIPE, pipesize=1024 * 1024) as p:
                return cursor.copy_expert(sql, p.stdin)
        else:
            opath = opath.with_suffix('.tab')
            with opath.open('w') as ofile:
                return cursor.copy_expert(sql, ofile, 1024 * 1024)
