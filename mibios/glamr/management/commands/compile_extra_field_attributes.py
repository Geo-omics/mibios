from pathlib import Path
import re

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.core.management.base import BaseCommand, CommandError

from mibios.glamr.models import Sample


TEMPLATE = 'extra_field_attributes.py.template'
DEFAULT_OUTPUT = 'extra_field_attributes.py'

# Name of input file under GLAMR_META_ROOT
UNITS_SHEET = 'Great_Lakes_Omics_Datasets.xlsx - metadata_units_and_notes.tsv'

# Required columns in input file:
FIELD_COL = 'django_field_name'
VERBOSE_COL = 'display_name'
INFOTXT_COL = 'info_text'
UNIT_COL = 'Units'


class Command(BaseCommand):
    help = 'compile the extra_module_attributes module'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-file',
            default=DEFAULT_OUTPUT,
            help=f'Path to output file, by default this is {DEFAULT_OUTPUT}',
        )

    def handle(self, *args, **options):
        infile = settings.GLAMR_META_ROOT / UNITS_SHEET
        data = {}
        with infile.open() as ifile:
            header = ifile.readline().rstrip('\n').split('\t')
            colindex = {}
            err_msg = []
            for i in [FIELD_COL, VERBOSE_COL, INFOTXT_COL, UNIT_COL]:
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
                row = [i.strip() for i in row]

                field_name = row[colindex[FIELD_COL]]
                attrs = {}

                if verbose_name := row[colindex[VERBOSE_COL]].strip():
                    attrs['verbose_name'] = verbose_name

                if info := row[colindex[INFOTXT_COL]].strip():
                    attrs['info_text'] = info

                if unit := row[colindex[UNIT_COL]]:
                    """
                    Can be either/or a real unit or some "-quoted text in
                    either order, e.g.:
                        fb
                        "foo bar"
                        "foo bar" fb
                        fb "foo bar"
                    """
                    if unit.startswith('"'):
                        if m := re.match(r'^"([^"]*)"(.*)$', unit):
                            pseudo_unit, proper_unit = m.groups()
                        else:
                            raise CommandError(
                                f'failed parsing unit at line {lineno}: {unit}'
                            )
                    elif unit.endswith('"'):
                        if m := re.match(r'^([^"]*)"([^"]*)"$', unit):
                            proper_unit, pseudo_unit = m.groups()
                        else:
                            raise CommandError(
                                f'failed parsing unit at line {lineno}: {unit}'
                            )
                    else:
                        proper_unit = unit
                        pseudo_unit = ''
                    if proper_unit := proper_unit.strip():
                        attrs['unit'] = proper_unit
                    if pseudo_unit := pseudo_unit.strip('"').strip():
                        attrs['pseudo_unit'] = pseudo_unit

                if not field_name:
                    print(f'[WARNING] django fieldname missing: line:{lineno} '
                          f'{row[0]=}')
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

        unlisted = [
            i.name for i in Sample._meta.get_fields()
            if i.name not in data and not i.many_to_one and not i.one_to_many
        ]
        if unlisted:
            print('[Notice] Fields not listed in units sheet:')
            for i in unlisted:
                print('   ', i)

        app_conf = apps.get_app_config(Sample._meta.app_label)
        templ_path = Path(app_conf.path) / TEMPLATE
        if not templ_path.is_file():
            raise CommandError(f'template file not found: {templ_path}')

        out_path = Path(options['output_file'])

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
