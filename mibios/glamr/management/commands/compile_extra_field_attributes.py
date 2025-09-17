from ast import literal_eval
from pathlib import Path
import re

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.core.management.base import BaseCommand, CommandError

from mibios.glamr.models import Sample
from mibios.omics.models import SeqSample


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
        biosample_data = {}
        seqsample_data = {}
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
                    biosample_field = Sample._meta.get_field(field_name)
                except FieldDoesNotExist:
                    biosample_field = None

                try:
                    seqsample_field = SeqSample._meta.get_field(field_name)
                except FieldDoesNotExist:
                    seqsample_field = None

                if biosample_field is None and seqsample_field is None:
                    print(f'[WARNING]: input line {lineno}: no such field: '
                          f'{field_name}, skipping offending line:\n'
                          f'{line.rstrip()}')
                    continue
                elif biosample_field is not None:
                    if seqsample_field is not None:
                        raise CommandError(
                            f'Is a field in both models: {field_name}'
                        )
                    # it's a (Bio)Sample field
                    if field_name in biosample_data:
                        raise CommandError(
                            f'duplicate biosample field name: {field_name}'
                        )
                    biosample_data[field_name] = attrs
                else:
                    # it's a SeqSample field
                    if field_name in seqsample_data:
                        raise CommandError(
                            f'duplicate seqsample field name: {field_name}'
                        )
                    seqsample_data[field_name] = attrs

        # check unlisted for (Bio)Sample only
        unlisted = [
            i.name for i in Sample._meta.get_fields()
            if i.name not in biosample_data
            and not i.many_to_one and not i.one_to_many
        ]
        if unlisted:
            print('[Notice] (Bio)Sample fields not listed in units sheet:')
            for i in unlisted:
                print('   ', i)

        app_conf = apps.get_app_config(Sample._meta.app_label)
        templ_path = Path(app_conf.path) / TEMPLATE
        if not templ_path.is_file():
            raise CommandError(f'template file not found: {templ_path}')

        out_path = Path(options['output_file'])

        with templ_path.open() as templ, out_path.open('w') as ofile:
            found_bio = False
            found_seq = False
            for line in templ:
                if line.strip().startswith('#') and 'PLACEHOLDER' in line:
                    if 'BIOSAMPLE' in line:
                        data = biosample_data
                        found_bio = True
                    elif 'SEQSAMPLE' in line:
                        data = seqsample_data
                        found_seq = True
                    else:
                        data = None

                    if data:
                        for field_name, attrs in data.items():
                            attrs_str = repr(attrs)
                            try:
                                # Unsure if code injection can still happen
                                # after repr() so check that this thing
                                # evaluates to a literal
                                literal_eval(attrs_str)
                            except Exception as e:
                                raise CommandError(
                                    f'code injection check tripped: '
                                    f'{e.__class__.__name__}: {e} '
                                    f'{field_name=} {attrs=}'
                                )
                            # write literal dict (field name to attr dict)
                            # items, single indentation:
                            ofile.write(f"    '{field_name}': {attrs_str},\n")
                        continue
                # copy from template
                ofile.write(line)

            if not found_bio or not found_seq:
                raise CommandError('templte is missing a placeholder')

        print(f'[OK] Saved as {out_path} --- Move the file into the correct '
              f'package directory, e.g. to:\n{app_conf.path}/')
