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

# The expected header of input file:
INPUT_HEADER = [
    'Column_name',
    VERBOSE_COL,
    UNIT_COL,
    'MIMARKS compliant',
    'MIMARKS required',
    FIELD_COL,
    'Notes',
]


class Command(BaseCommand):
    help = 'compile the extra_module_attributes module'

    def handle(self, *args, **options):
        infile = settings.GLAMR_META_ROOT / UNITS_SHEET
        data = {}
        with infile.open() as ifile:
            header = ifile.readline().rstrip('\n').split('\t')
            if header != INPUT_HEADER:
                print(f'WARNING: unexpected input file header:\n'
                      f'expected: {INPUT_HEADER}\n'
                      f'     got: {header}')
            try:
                field_col_i = header.index(FIELD_COL)
                verb_col_i = header.index(VERBOSE_COL)
                unit_col_i = header.index(UNIT_COL)
            except ValueError:
                raise CommandError(
                    'ERROR: failed to find at least one required inputfile '
                    'column'
                )

            for lineno, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                field_name = row[field_col_i]

                attrs = {}
                if row[verb_col_i]:
                    attrs['verbose_name'] = row[verb_col_i]
                unit = row[unit_col_i]
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

        with templ_path.open() as templ, open(OUTPUT, 'w') as ofile:
            for line in templ:
                if line.strip().startswith('# PLACEHOLDER'):
                    for field_name, attrs in data.items():
                        # write literal dict (field name to attr dict) items,
                        # single indentation:
                        ofile.write(f"    '{field_name}': {attrs},\n")  # noqa: E501
                else:
                    ofile.write(line)

        print(f'[OK] Saved to {OUTPUT} in current dir, you will need to '
              f'manually move the file into the correct package directory, '
              f'e.g.: {app_conf.path}/')
