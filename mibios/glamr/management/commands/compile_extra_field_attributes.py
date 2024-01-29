from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.core.management.base import BaseCommand, CommandError

from mibios.glamr.models import Sample


TEMPLATE = 'extra_field_attributes.py.template'
OUTPUT = 'extra_field_attributes.py'

# Name of input file under GLAMR_META_ROOT
UNITS_SHEET = 'Great_Lakes_Omics_Datasets.xlsx - metadata_units_and_notes.tsv'

# Required columns in input file:
FIELD_COL = 'django_field_name'
VERBOSE_COL = 'verbose_name'
UNIT_COL = 'Units'


class Command(BaseCommand):
    help = 'compile the extra_module_attributes module'

    def add_arguments(self, parser):
        parser.add_argument('--output-file', help='Path to output file.')

    def handle(self, *args, **options):
        infile = settings.GLAMR_META_ROOT / UNITS_SHEET
        data = {}
        with infile.open() as ifile:
            header = ifile.readline().rstrip('\n').split('\t')
            colindex = {}
            err_msg = []
            for i in [FIELD_COL, VERBOSE_COL, UNIT_COL]:
                try:
                    colindex[i] = header.index(i)
                except ValueError:
                    err_msg.append(f'ERROR: no such column: {i}')

            if err_msg:
                raise CommandError(
                    '\n' + '\n'.join(err_msg) + '\n'
                    + f'ERROR: Bad header in input file: {infile}'
                )

            for lineno, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                field_name = row[colindex[FIELD_COL]]

                attrs = {}
                if row[colindex[VERBOSE_COL]]:
                    attrs['verbose_name'] = row[colindex[VERBOSE_COL]]
                unit = row[colindex[UNIT_COL]]
                if unit:
                    if unit.startswith('"') and unit.endswith('"'):
                        attrs['pseudo_unit'] = unit.strip('"')
                    else:
                        attrs['unit'] = unit

                if not field_name:
                    # ignore this row
                    continue

                try:
                    Sample._meta.get_field(field_name)
                except FieldDoesNotExist:
                    print(f'[WARNING]: input line {lineno}: no such field: '
                          f'{field_name}, skipping offending line:\n'
                          f'{line.rstrip()}')
                    continue

                if field_name in data:
                    raise CommandError(
                        f'ERROR: duplicate django field name in input file: '
                        f'{field_name}'
                    )

                data[field_name] = attrs

        app_conf = apps.get_app_config(Sample._meta.app_label)
        templ_path = Path(app_conf.path) / TEMPLATE
        if not templ_path.is_file():
            raise CommandError(f'template file not found: {templ_path}')

        out_path = Path(options.get('output_file', OUTPUT))

        with templ_path.open() as templ, out_path.open('w') as ofile:
            for line in templ:
                if line.strip().startswith('# PLACEHOLDER'):
                    for field_name, attrs in data.items():
                        # write literal dict (field name to attr dict) items,
                        # single indentation:
                        ofile.write(f"    '{field_name}': {attrs},\n")
                else:
                    ofile.write(line)

        print(f'[OK] Saved as {out_path} --- Move the file into the correct '
              f'package directory, e.g. to:\n{app_conf.path}/')
