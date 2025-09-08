from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory
import subprocess

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from mibios.glamr.models import Sample


AUTH_MODELS = (
    'user', 'group', 'permission', 'group_permissions', 'user_groups',
    'user_user_permissions',
)

GLAMR_MODELS = (
    'Reference', 'Dataset', 'Sample', 'AboutInfo', 'Credit',
    'dataset_restricted_to', 'dataset_references',
)

OMICS_SAMPLE_REL_MODELS = (
    'ReadAbundance', 'Bin', 'File', 'TaxonAbundance', 'Contig',
    'FuncAbundance', 'SampleTracking',
)

TAX_MODELS = ('Division', 'Gencode', 'TaxNode', 'TaxName', 'MergedNodes',
              'DeletedNode')


class Command(BaseCommand):
    help = 'Create a dump of a test database from production or staging'

    def add_arguments(self, parser):
        parser.add_argument(
            'sample',
            nargs='+',
            help='Sample IDs',
        )
        parser.add_argument(
            '-o', '--output-file',
            default='test_dump',
            help='Name of output file, without suffix',
        )

    def handle(self, *args, **options):
        self.options = options
        sample_qs = Sample.objects.filter(sample_id__in=options['sample'])
        self.samples = list(sample_qs)

        ids = set((i.sample_id for i in self.samples))
        for i in options['sample']:
            if i not in ids:
                raise CommandError(f'No such sample: {i}')
        del ids

        out_arg = Path(options['output_file'])
        out_dir = out_arg.parent
        base_name = out_arg.name.removesuffix('.tar.zst')
        with TemporaryDirectory() as tmpd:
            self.tmp = Path(tmpd)
            tarfile = self.tmp / (base_name + '.tar')
            with TarFile.open(tarfile, mode='w') as tarf:
                self.tarf = tarf
                self.dump_all()

            outpath = out_dir / (base_name + '.tar.zst')
            subprocess.run(
                # --force to not prompt when overwriting existing file
                ['zstd', '--force', '-o', str(outpath), str(tarfile)],
                check=True,
            )

    def dump_all(self):
        """
        Dump tables in order.  Earlier tables must not have FKs referencing
        later tables (this will bite when loading the data.)  Requires the
        postgresql database backend.
        """
        # contentypes
        self.dump(model=apps.get_model('contenttypes', 'contenttype'))

        # all auth models
        for i in AUTH_MODELS:
            model = apps.get_model('auth', i)
            self.dump(model=model)

        # glamr model get dumped whole
        for i in GLAMR_MODELS:
            model = apps.get_model('glamr', i)
            self.dump(model=model)

        # get whole taxonomy
        for i in TAX_MODELS:
            model = apps.get_model('ncbi_taxonomy', i)
            self.dump(model=model)

        # uniref100s related to samnples (via ReadAbundance)
        UniRef100 = apps.get_model('umrad', 'UniRef100')
        ur100_qs = UniRef100.objects\
            .filter(abundance__sample__in=self.samples)\
            .distinct()

        self.dump(queryset=ur100_qs)

        # omics models related to our samples
        for i in OMICS_SAMPLE_REL_MODELS:
            model = apps.get_model('omics', i)
            qs = model.objects.filter(sample__in=self.samples)
            self.dump(queryset=qs)

        # The Bin<->Contig m2m relation
        Bin = apps.get_model('omics', 'bin')
        Through = Bin._meta.get_field('contigs').remote_field.through
        qs = Through.objects.filter(bin__sample__in=self.samples)
        self.dump(queryset=qs)

    def dump(self, *, queryset=None, model=None):
        """
        Dump data from one table
        """
        if queryset is None and model is None:
            raise ValueError('either queryset or model must be provided')

        if model:
            table_name = model._meta.db_table
            sql = f'COPY {table_name} TO STDOUT'
        else:
            # queryset given
            table_name = queryset.model._meta.db_table
            sql = f'COPY ( {queryset.query} ) TO STDOUT'

        out = self.tmp / f'{table_name}.dump.text'
        print(f'{out.name:<37}', end=' ', flush=True)
        if queryset is None:
            print('[whole]', end=' ', flush=True)
        else:
            print('[ part]', end=' ', flush=True)

        if queryset is not None:
            # NOTE: This is fragile.  The dump happens with columns ordered as
            # specified by the query.  Hoping this will always be the models's
            # concrete fields in that order.  By contrast, the whole table
            # dumps seems to have columns ordered the way they are printed by
            # the \d {table_name} command and there is hoping that that won't
            # change between databases.
            col_file = self.tmp / out.with_suffix('.cols')
            with col_file.open('w') as ofile:
                for i in queryset.model._meta.concrete_fields:
                    ofile.write(f'{i.column}\n')
            self.tarf.add(col_file, arcname=col_file.name)

        with open(out, 'w') as ofile:
            cur = connection.cursor()
            try:
                cur.copy_expert(sql, ofile)
            except KeyboardInterrupt:
                print('\n^C', end=' ', flush=True)
                cur.connection.cancel()
                print('cancel', end=' ', flush=True)
                cur.connection.rollback()
                print('rollback', end=' ', flush=True)
                cur.connection.close()
                print('close')
                raise
            else:
                cur.close()

            count = cur.rowcount

        size_gb = out.stat().st_size / 2**30
        print(f'{count:>9} rows [{size_gb:5.1f}G]', end=' ', flush=True)
        self.tarf.add(out, arcname=out.name)
        print('[O', end='', flush=True)
        out.unlink()
        print('K]')
