from collections import Counter
from decimal import Decimal
from itertools import cycle

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from mibios.glamr.models import Sample


# Name of input file under GLAMR_META_ROOT
SAMPLE_SHEET = 'Great_Lakes_Omics_Datasets.xlsx - samples.tsv'


class Command(BaseCommand):
    help = 'Show statistics of sample sheet column data.'

    default_input_file = settings.GLAMR_META_ROOT / SAMPLE_SHEET

    def add_arguments(self, parser):
        parser.add_argument(
            'columns',
            metavar='column',
            nargs='*',
            help='one or more column or field identifier, optional',
        )
        parser.add_argument(
            '-f', '--input-file',
            default=self.default_input_file,
            help='Input file, expected to be .tsv downloaded '
                 'Great_Lakes_Omics_Datasets samples sheet.  The default is to'
                 f' use "{self.default_input_file}" .',
        )
        grp = parser.add_mutually_exclusive_group()
        grp.add_argument(
            '--list',
            action='store_true',
            help='List all distinct values instead of giving a summary.',
        )
        grp.add_argument(
            '--type-check',
            action='store_true',
            help='Run in type check mode.',
        )

    def handle(self, *args, **options):
        self.counts = {}

        # Read data from file
        self.data = self.read_data(options['input_file'])
        self.code = {
            name: code
            for name, code in zip(self.data, self.sheet_col_code_iter())
        }

        # Import Sample model stuff
        spec = Sample.loader.spec
        spec.setup(Sample.loader)

        self.col_fields = {
            i: j
            for i, j in zip(spec.col_names, spec.fields)
            if isinstance(i, str)  # skip technical columns
        }
        field_names = {
            i.name: col
            for i, col in zip(spec.fields, spec.col_names)
            if isinstance(col, str)
        }

        if not options['columns']:
            # Compare sheet file with Sample model unless columns are given
            for i in self.data:
                if i not in self.col_fields:
                    print(f'missing from spec: {i} ({self.code[i]})')
            for i in self.col_fields:
                if i not in self.data:
                    print(f'spec entry missing in sample sheet: {i} / '
                          f'{self.col_fields[i]}')

        # Compile column todo list
        if options['columns']:
            todo = []
            for i in options['columns']:
                if i in self.data:
                    col = i
                elif i in field_names:
                    col = field_names[i]
                else:
                    raise CommandError(
                        f'does not match a column or field name: {i}'
                    )
                if col not in todo:
                    todo.append(col)
        else:
            todo = self.data.keys()

        # Process data and print results
        for i in todo:
            self.do_accounting(i)
            if options['list']:
                self.print_distinct_values(i)
            elif options['type_check']:
                if i in self.col_fields:
                    self.run_type_check(i)
                else:
                    if options['columns']:
                        # was requested so err out
                        raise CommandError(
                            f'can not type check: no associated field for {i}'
                        )
                    else:
                        # skip this columns
                        continue
            else:
                self.print_summary(i)

    def read_data(self, inputfile):
        data = {}
        with open(inputfile) as ifile:
            head = ifile.readline().rstrip('\n').split('\t')
            for i in head:
                data[i] = []

            if len(data) < len(head):
                raise CommandError('duplicate column header')

            for lineno, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                if len(row) != len(head):
                    raise CommandError(f'row length differs from header at '
                                       f'line {lineno}')

                for colname, value in zip(head, row):
                    data[colname].append(value)
        return data

    def do_accounting(self, colname):
        blanks = Counter()
        ints = Counter()
        decs = Counter()
        strings = Counter()

        for i in self.data[colname]:
            if i in ['', 'NF', 'NA', 'N/A']:
                blanks[i] += 1
                continue

            if i.isdecimal():
                try:
                    i = int(i)
                except Exception:
                    pass
                else:
                    ints[i] += 1
                    continue

            try:
                i = Decimal(i)
            except Exception:
                pass
            else:
                decs[i] += 1
                continue

            strings[i] += 1

        self.counts[colname] = counts = {}
        counts['blanks'] = blanks
        counts['ints'] = ints
        counts['decs'] = decs
        counts['strings'] = strings

    def print_summary(self, colname):
        """ print summary of column stats """
        counts = self.counts[colname]
        blanks = counts['blanks']
        ints = counts['ints']
        decs = counts['decs']
        strings = counts['strings']

        if field := self.col_fields.get(colname):
            field_name = field.name
        else:
            field_name = 'no field'

        out = (f'{colname} <{self.code[colname]}> [{field_name}]:')

        if blanks:
            if ints or decs or strings:
                mod = ''
            else:
                mod = 'all '
            out += f' {mod}{blanks.total()} blanks'

        if ints:
            out += f'\n       integers: {self.counter_stats(ints)}'

        if decs:
            out += f'\n       decimals: {self.counter_stats(decs)}'
            maxdigits, places = self.decimal_param_stats(colname)
            out += f'\n       dec params: max digits: {maxdigits} // '
            out += 'decimal places: ' + ','.join((str(i) for i in places))

        if strings:
            out += f'\n           text: {self.counter_stats(strings)}'

        print(out)

    def print_distinct_values(self, colname):
        """ print listing of distinct values """
        counts = self.counts[colname]

        data = sorted(counts['blanks'].items())
        data += sorted((counts['ints'] | counts['decs']).items())
        data += sorted(counts['strings'].items())
        print(f'{colname} ({self.code[colname]}):')
        for val, count in data:
            print(f'    {count:>5}   {val}')

    @classmethod
    def counter_stats(cls, counter):
        """ helper to get statistics for a Counter """
        total = counter.total()
        distinct = len(counter)
        listing = sorted(counter)
        if len(listing) > 5:
            listing = [listing[0], listing[1], ..., listing[-2], listing[-1]]
        listing = ', '.join((cls.format_value(i) for i in listing))

        out = f'{total:>5}'
        if distinct == 1:
            out += ' [all the same]'
        elif total > distinct:
            most_val, most_count = counter.most_common()[0]
            out += f' [distinct:{distinct} '
            out += f'most common ({most_count}): '
            out += f'{cls.format_value(most_val)}]'
        else:
            out += ' [all distinct]'

        out += ' => ' + listing
        return out

    def decimal_param_stats(self, colname):
        """ Extract precision parameters needed for DecimalField """
        counts = self.counts[colname]
        ints = counts['ints']
        decs = counts['decs']
        values = [str(i).removeprefix('-') for i in list(ints) + list(decs)]
        # max whole digits in a value:
        maxwhole = max((len(i.partition('.')[0]) for i in values))
        # collect numbers of fractional places occurring in values
        places = sorted(set((len(i.partition('.')[2]) for i in values)))
        maxdigits = maxwhole + max(places)
        return maxdigits, places

    @classmethod
    def format_value(cls, value):
        """ helper to format value depending on type """
        if value is ...:
            return '...'
        if isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    @classmethod
    def sheet_col_code_iter(cls):
        """ Generate spreadsheet column IDs """
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        state = []
        current = []
        while True:
            pos = 0
            while True:
                if len(state) == pos:
                    state.append(cycle(abc))
                    current.append(None)

                value = next(state[pos])
                overflow = current[pos] == abc[-1]
                current[pos] = value
                if overflow:
                    # advance next position
                    pos += 1
                    continue
                else:
                    # leave other positions as-is
                    break

            yield ''.join(reversed(current))

    def run_type_check(self, colname):
        TEXTLIKE = ['CharField', 'ForeignKey', 'OneToOneField', 'TextField']
        POSINT = ['PositiveIntegerField', 'PositiveSmallIntegerField']

        counts = self.counts[colname]
        ints = counts['ints']
        decs = counts['decs']
        strings = counts['strings']
        field = self.col_fields[colname]
        ftype = field.get_internal_type()

        if decs or ftype == 'DecimalField':
            maxdigits, places = self.decimal_param_stats(colname)

        id_str = f'{colname} <{self.code[colname]}> [{field.name}]:'
        if ftype in TEXTLIKE:
            if not strings:
                if ints and not decs:
                    print(f'{id_str} {ftype} has only integer data')
                elif not ints and decs:
                    print(f'{id_str} {ftype} has only decimal data')
                if ints and decs:
                    print(f'{id_str} {ftype} has only numeric (ints+decimal)'
                          f' data')
                if decs:
                    print(f'{id_str} decimal params: {maxdigits}, {places}')

        elif ftype in POSINT:
            if strings:
                print(f'{id_str} {ftype} has strings')
            if decs:
                print(f'{id_str} {ftype} has decimals, params: {maxdigits}, '
                      f'{places}')
        elif ftype == 'DecimalField':
            params_data = (maxdigits, places[-1])
            params_field = (field.max_digits, field.decimal_places)

            if params_data != params_field:
                print(f'{id_str} decimal params (max_digits, decimal_places) '
                      f'mismatch: '
                      f'field:{params_field}  data: {maxdigits}, {places}')
            if strings:
                print(f'{id_str} {ftype} has strings')
            if ints and not decs:
                print(f'{id_str} {ftype} has integers but no decimals')
