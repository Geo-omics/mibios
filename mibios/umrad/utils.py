from datetime import datetime
from functools import partial, wraps
from inspect import signature
from io import UnsupportedOperation
from itertools import chain, zip_longest
from operator import length_hint
import os
from pathlib import Path
from threading import local, Timer
from string import Formatter
import sys

import pandas

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import router, transaction
from django.db.models import Q

# Workaround for weird pandas/xlrd=1.2/defusedxml combination runtime issue,
# we'll get an AttributeError: 'ElementTree' object has no attribute
# 'getiterator' inside xlrd when trying to pandas.read_excel().  See also
# https://stackoverflow.com/questions/64264563
import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True


thread_data = local()
thread_data.timer = None


def get_last_timer():
    return thread_data.timer


class InputFileError(Exception):
    """
    malformed line in input file

    We may expect this error and may tolerate it and skip the offending line
    """
    def __init__(self, *args):
        args = [f'{type(i).__name__}: {i}' if isinstance(i, Exception) else i
                for i in args]
        super().__init__(*args)


class ReturningGenerator:
    """
    A wrapper to catch return values of generators

    Usage:
    def g(*args):
        ...
        yield ...
        ...
        return x

    g = ReturningGenerator(g())
    for i in g:
        do_stuff()
    foo = g.value

    An AttributeError will be raised if one attempts to access the return value
    before the generator is exhausted.
    """
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        self.value = yield from self.generator


class RepeatTimer(Timer):
    """
    Run given function repeatedly and wait a given interval between invocations

    From an answer to stackoverflow.com/questions/12435211
    """
    def __init__(self, *args, owner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner
        # let main thread exit if we're forgotten and never stop ticking
        self.daemon = True

    def start(self):
        thread_data.timer = self
        super().start()

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class ProgressPrinter():
    """
    A simple timer-based printer of progress or counts

    How to use:

    pp = ProgressPrinter('{progress} foo done')
    count = 0
    for i in some_iterator():
        do_stuff()
        count += 1
        pp.update(count)
    pp.finished()

    This will print "<n> foo done" once per second to the terminal and print
    the final count after the for loop ends. The internal timer will keep
    restarting as long as the update() method is called with changing progress
    counts.

    After the timer has stopped, even after calling finish() the progress
    printing can be resumed by calling update() with a different state than the
    last one.
    """
    DEFAULT_TEMPLATE = '{progress}'
    DEFAULT_INTERVAL = 1.0  # seconds

    def __init__(
            self,
            template=DEFAULT_TEMPLATE,
            interval=DEFAULT_INTERVAL,
            output_file=None,
            show_rate=True,
            length=None,
    ):
        if interval <= 0:
            raise ValueError('interval must be greater than zero')

        self.template, self.template_var = self._init_template(template)
        self.interval = interval
        self.output_file = output_file or sys.stdout
        self.show_rate = show_rate
        self.to_terminal = self.output_file.isatty()
        self.length = length
        self.it = None
        self.timer = None
        # start metering here, assuming inc() calls commence soon:
        self.reset_state()
        self.timer.start()

    def __call__(self, it):
        self.it = it
        self.reset_length_info()
        for elem in it:
            yield elem
            self.inc()
        self.it = None
        self.finish()

    def reset_state(self):
        """
        reset the variable state

        Must be called before inc().  Will start the timer and progress
        metering and printing.
        """
        self.checkpoint = False
        self.reset_timer()
        self.total = 0
        self.old_total = 0
        self.max_width = 0
        self.time_zero = datetime.now()
        self.time_start = self.time_zero
        self.time_end = None
        self.duration = None
        self.reset_length_info()

    def reset_length_info(self):
        """
        get length if possible
        """
        self._length = None
        self.file_size = None
        if self.length is None:
            try:
                self.file_size = os.stat(self.it.fileno()).st_size
            except Exception:
                self.file_size = None

                hint = length_hint(self.it)
                if hint > 0:
                    # 0 is the default in case there is no length or length
                    # hint, it seems we can't tell this from an actual length
                    # of zero
                    self._length = hint
        else:
            self._length = self.length

    def reset_timer(self):
        """
        Initializes or resets the timer but does not start() it.
        """
        if self.timer is not None:
            self.timer.cancel()
        del self.timer
        self.timer = RepeatTimer(self.interval, self._ring, owner=self)

    def _init_template(self, template):
        """
        set up template

        We support templates with zero or one formatting fields, the single
        field may be named or anonymous.
        """
        fmt_vars = [
            i[1] for i
            in Formatter().parse(template)
            if i[1] is not None
        ]
        if len(fmt_vars) == 0:
            template = '{} ' + template
            template_var = None
        elif len(fmt_vars) == 1:
            # keep tmpl as-is
            if fmt_vars[0] == '':
                template_var = None
            else:
                template_var = fmt_vars[0]
        else:
            raise ValueError(f'too many format fields in template: {fmt_vars}')

        return template, template_var

    def inc(self, step=1):
        """
        increment progress

        Turn on time if needed
        """
        self.total += step

        if self.checkpoint:
            self.time_end = datetime.now()
            self.duration = (self.time_end - self.time_start).total_seconds()
            self.print_progress()
            self.old_total = self.total
            self.time_start = self.time_end
            self.checkpoint = False

    def finish(self):
        """ Stop the timer and print a final result """
        total_seconds = (datetime.now() - self.time_zero).total_seconds()
        avg_txt = (f'(total: {total_seconds:.1f}s '
                   f'avg: {self.total / total_seconds:.1f}/s)')
        self.print_progress(avg_txt=avg_txt, end='\n')  # print with totals/avg
        self.reset_state()

    def _ring(self):
        """ Run by timer """
        self.checkpoint = True

    def estimate(self):
        """
        get momentary percentage and estimated finish time

        Returns None if we havn't made any progress yet or we don't know the
        length of the iterator.
        """
        if self.total == 0:
            # too early for estimates (and div by zero)
            return

        if self.file_size is not None and self.file_size > 0:
            # best effort to get stream position, hopefully in bytes
            try:
                pos = self.it.tell()
            except Exception:
                # for iterating text io tell() is diabled?
                # (which doesn't seem to be documented?)
                try:
                    pos = self.it.buffer.tell()
                except Exception:
                    return
            frac = pos / self.file_size

        elif self._length is not None and self._length > 0:
            frac = self.total / self._length
        else:
            # no length info
            return

        total_seconds = (self.time_end - self.time_zero).total_seconds()
        remaining_seconds = total_seconds / frac - total_seconds
        return frac, remaining_seconds

    def print_progress(self, avg_txt='', end=''):
        """ Do the progress printing """
        if self.template_var is None:
            txt = self.template.format(self.total)
        else:
            txt = self.template.format(**{self.template_var: self.total})

        if avg_txt:
            # called by finish()
            txt += ' ' + avg_txt
        elif self.show_rate:
            try:
                rate = (self.total - self.old_total) \
                    / (self.time_end - self.time_start).total_seconds()
            except Exception:
                # unlikely zero div or some None values
                pass
            else:
                est = self.estimate()
                if est is None:
                    # just the rate
                    txt += f' ({rate:.0f}/s)'
                else:
                    frac, remain = est
                    txt += f' ({frac:.0%} rate:{rate:.0f}/s remaining:{remain:.0f}s)'  # noqa:E501

        self.max_width = max(self.max_width, len(txt))
        txt = txt.ljust(self.max_width)

        if self.to_terminal:
            txt = '\r' + txt

        print(txt, end=end, flush=True, file=self.output_file)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into non-overlapping groups of n elements

    This is the grouper from the stdlib's itertools' recipe book.
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def chunker(iterable, n):
    """
    Group iterable in chunks of equal size, except possibly for the last chunk
    """
    sentinel = object()
    for grp in grouper(iterable, n, fillvalue=sentinel):
        # grp is a n-tuple
        if grp[-1] is sentinel:
            yield tuple(i for i in grp if i is not sentinel)
        else:
            yield grp


class SpecError(Exception):
    """ raised on invalid InputFileSpec """
    pass


class InputFileSpec:
    IGNORE_COLUMN = object()
    SKIP_ROW = object()
    CALC_VALUE = object()
    NO_HEADER = object()

    empty_values = []
    """
    A list of input-file-wide extra empty values.  For the purpose of loading
    the data these are used in addition to each field's empty_values attribute.
    """

    def __init__(self, *column_specs):
        self._spec = column_specs or None

        # set by setup():
        self.model = None
        self.loader = None
        self.file = None
        self.has_header = None
        self.fk_attrs = {}
        self.fkmap_filters = {}
        self.pre_load_hook = []

    def setup(self, loader, column_specs=None, file=None):
        """
        Setup method to be called once before loading data

        Intended to be called automatically by the loader.  Should be
        idempotent when called repeatedly.
        """
        self.loader = loader
        self.model = loader.model
        if column_specs is None:
            column_specs = self._spec

        if not column_specs:
            raise SpecError('at least one column needs to be declared')

        if file is None:
            file = self.loader.get_file()
        if isinstance(file, str):
            file = Path(file)
        self.file = file

        self.empty_values += self.loader.empty_values

        col_names = []  # all row column headers, as in file
        col_index = []  # index of actually used columns
        keys = []
        field_names = []
        fields = []
        prepfuncs = []

        cur_col_index = None  # for non-header input, defined by order in spec
        for spec_line in column_specs:
            # TODO: re-write as match statement?
            # FIXME: two-str-items format can be <col> <field> OR <field> <fun>
            # with no super easy way to distinguish them
            if not isinstance(spec_line, tuple):
                # no-header-simple-format
                spec_line = (spec_line, )

            if self.has_header is None:
                # detect header presence from first spec piece
                if len(spec_line) == 1:
                    self.has_header = False
                elif len(spec_line) == 2 and callable(spec_line[1]):
                    self.has_header = False
                else:
                    self.has_header = True

            if not self.has_header:
                spec_line = (self.NO_HEADER, *spec_line)

            colname, key, *prepfunc = spec_line

            if colname is self.CALC_VALUE:
                if key is None:
                    raise SpecError('require key (field name) for for which'
                                    'to calculate a value')

                if not self.has_header:
                    col_index.append(self.CALC_VALUE)
            else:
                # current spec item is for a column in input
                if not self.has_header:
                    # add 0-based numerical index for column name
                    # these have to be counted here
                    if cur_col_index is None:
                        cur_col_index = 0
                    else:
                        cur_col_index += 1

                if key in (None, self.IGNORE_COLUMN):
                    # ignore this column
                    continue

                if not self.has_header:
                    col_index.append(cur_col_index)

            if not isinstance(key, str):
                raise SpecError(f'expecting a str: {key=} in {spec_line=}')

            if '.' in key:
                field_name, _, attr = key.partition('.')
                self.fk_attrs[field_name] = attr
            else:
                field_name = key

            col_names.append(colname)
            keys.append(key)

            try:
                field = self.model._meta.get_field(field_name)
            except FieldDoesNotExist as e:
                raise SpecError(f'bad spec line: {spec_line}: {e}') from e

            fields.append(field)
            field_names.append(field_name)

            if len(prepfunc) == 0:
                # no pre-proc method set, add auto-magic stuff as-needed here
                if field.choices:
                    # automatically attach prep method for choice fields
                    prepfunc = self.loader.get_choice_value_prep_method(field)
            elif len(prepfunc) > 1:
                raise SpecError(f'too many items in spec for {colname}/{key}')
            elif len(prepfunc) == 1 and prepfunc[0] is None:
                # pre-proc method explicitly set to None
                prepfunc = None
            else:
                # a non-None pre-proc method is given
                prepfunc = prepfunc[0]
                if isinstance(prepfunc, str):
                    prepfunc_name = prepfunc
                    # getattr gives us a bound method:
                    prepfunc = getattr(loader, prepfunc_name)
                    if not callable(prepfunc):
                        raise SpecError(
                            f'not the name of a {self.loader} method: '
                            f'{prepfunc_name}'
                        )
                elif callable(prepfunc):
                    # Assume it's a function that takes the loader as
                    # 1st arg.  We get this when the previoudsly
                    # delclared method's identifier is passed directly
                    # in the spec's declaration.
                    prepfunc = partial(prepfunc, self.loader)
                else:
                    raise SpecError(f'not a callable: {prepfunc}')

            prepfuncs.append(prepfunc)

        self.col_names = col_names
        self.col_index = col_index
        self.keys = keys
        self.field_names = field_names
        self.fields = tuple(fields)
        self.prepfuncs = tuple(prepfuncs)

        if callable(self.pre_load_hook):
            self.pre_load_hook = [self.pre_load_hook]
        try:
            for i in self.pre_load_hook:
                if not callable(i):
                    raise TypeError(f'not callable: {i}')
        except TypeError as e:
            # get here also if we can't iterate over per_load_hook
            msg = 'pre_load_hook should be a list of callables'
            raise SpecError(msg) from e

    def __len__(self):
        return len(self.keys)

    def iterrows(self):
        """
        A generator for the records/rows of the input file

        This method must be implemented by inheriting classes.  The generator
        should yield a sequence of items for each record or row.
        """
        raise NotImplementedError

    def row_data(self, row):
        """
        Generate a row as tuples (field, func, value)

        Blank/empty values will be set to None here.  Extra items for
        calculated field values are added.

        :param list row: A list of str
        """
        it = zip(self.fields, self.prepfuncs, self.col_names, self.col_index)
        for field, fn, col_name, col_i in it:
            if col_name is self.CALC_VALUE:
                if fn is None:
                    # No method was provided in spec, so we let them skip this
                    # one and trust that this field was or will be set via some
                    # other field's processing.
                    value = self.IGNORE_COLUMN
                else:
                    # fn will calculate value
                    value = None
            else:
                try:
                    value = row[col_i]
                except IndexError as e:
                    raise InputFileError(
                        f'row too short: no element with index {col_i} for '
                        f'field {field} / column {col_name} {row=}'
                    ) from e
                if value in self.empty_values or value in field.empty_values:
                    value = None
            yield (field, fn, value)

    def row2dict(self, row_data):
        """ turn result of row_data() into a dict with field names as keys """
        return {field.name: val for field, _, val in row_data}


class CSVRowGenerator:
    """
    Generator over csv rows with cleanup

    Usage:
        spec = CSVSpec(...)
        for row in CSVRowGenerator(spec):
            ...

    This hides the complexity of supporting normal filesystem-based files as
    well as tempfiles or other things like StringIO.  Normal files will be open
    and closed.  Others will be left open and have seek(0) run so that they can
    be reused.  It is the loader's responsibility to clean those up.
    """
    def __init__(self, spec):
        self.spec = spec
        try:
            # open in case it's a str or Path
            self._file = open(self.spec.file)
        except TypeError:
            # assume it's already file-like, e.g. some tempfile
            self._file = self.spec.file
            self.keep_open = True
        else:
            print(f'File opened: {self._file.name}')
            self.keep_open = False
            try:
                os.posix_fadvise(self._file.fileno(), 0, 0,
                                 os.POSIX_FADV_SEQUENTIAL)
            except UnsupportedOperation:
                pass

        if self.spec.has_header:
            self.spec.process_header(self._file)

        self._it = self._generator()

    def _generator(self):
        """ the row generator with cleanup """
        try:
            for line in self._file:
                yield line.rstrip('\n').split(self.spec.sep)
        finally:
            if self.keep_open:
                # in case tempfile is read multiple times
                self._file.seek(0)
            else:
                self._file.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class CSV_Spec(InputFileSpec):
    def __init__(self, *column_specs, sep='\t'):
        super().__init__(*column_specs)
        self.sep = sep

    def setup(self, *args, sep=None, **kwargs):
        super().setup(*args, **kwargs)
        if sep is not None:
            self.sep = sep

    def process_header(self, file):
        """
        consumes and checks a single line of columns headers

        Calling this will set up the column_index that is needed to iterate
        over the rows.

        Overwrite this method if your file has a more complex layout.  Make
        sure this method consumes all non-data rows at the beginning of the
        file.  If the input file is read multiple times then process_header()
        is called each time, before iterating over the rows.
        """
        head = file.readline().rstrip('\n').split(self.sep)
        col_pos = {colname: pos for pos, colname in enumerate(head)}

        column_index = []
        for col in self.col_names:
            if col is self.CALC_VALUE:
                column_index.append(self.CALC_VALUE)
            else:
                try:
                    pos = col_pos[col]
                except KeyError:
                    raise InputFileError(
                        f'column "{col}" not found in header: {head}'
                    )

                column_index.append(pos)
        self.col_index = column_index

    def iterrows(self):
        """
        An iterator over the csv file's rows

        Also manages opening/closing of file if needed.  Will consume the
        header row if there is one.
        """
        return CSVRowGenerator(self)


class ExcelSpec(InputFileSpec):
    def __init__(self, *column_specs, sheet_name=None):
        super().__init__(*column_specs)
        self.sheet_name = sheet_name

    def get_dataframe(self):
        """ Return file as pandas.DataFrame """
        print(f'File opened: {self.file}')
        df = pandas.read_excel(
            str(self.file),
            sheet_name=self.sheet_name,
            # TODO: only read columns we need
            # usecols=...,
            na_values=self.empty_values,
            keep_default_na=False,
        )
        # turn NaNs (all the empty cells) into Nones
        # Else we'd have to tell load() to treat NaNs as empty
        df = df.where(pandas.notnull(df), None)
        return df

    def process_header(self, df):
        """
        Populate the column index
        """
        col_pos = {colname: pos for pos, colname in enumerate(df.columns)}

        column_index = []
        for col in self.col_names:
            if col is self.CALC_VALUE:
                column_index.append(col_pos)
            else:
                try:
                    pos = col_pos[col]
                except KeyError:
                    raise InputFileError(
                        f'column not found: {col=} in {col_pos=}'
                    )

                column_index.append(pos)
        self.col_index = column_index

    def iterrows(self):
        """
        An iterator over the table's rows, yields Pandas.Series instances

        :param pathlib.Path path: path to the data file
        """
        df = self.get_dataframe()
        self.process_header(df)  # this call has to be made somewhere
        for _, row in df.iterrows():
            yield row


class SizedIterator:
    """
    a wrapper to attach a known length to an iterator

    Example usage:

    g = (do_something(i) for i in a_list)
    g = SizedIterator(g, len(a_list))
    len(g)

    """
    def __init__(self, obj, length):
        self._it = iter(obj)
        self._length = length

    def __iter__(self):
        return self._it

    def __next__(self):
        return next(self._it)

    def __len__(self):
        return self._length


def siter(obj, length=None):
    """
    Return a sized iterator

    Convenience function for using the SizedIterator.  If length is not given
    then len(obj) must work.  Compare to the one-argument form of the built-in
    iter() function
    """
    if length is None:
        length = len(obj)
    return SizedIterator(obj, length)


def atomic_dry(f):
    """
    Replacement for @atomic decorator for Manager methods

    Supports dry_run keyword arg and calls set_rollback appropriately and
    coordinates the db alias in case we have multiple databases.  This assumes
    that the decorated method has self.model available, as it is the case for
    managers, and that if write operations are used for other models then those
    must run on the same database connection.
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        dbalias = router.db_for_write(self.model)
        with transaction.atomic(using=dbalias):
            if 'dry_run' in kwargs:
                dry_run = kwargs['dry_run']
                if dry_run or 'dry_run' not in signature(f).parameters:
                    # consume dry_run kw if True as to avoid nested rollback
                    # but pass on a dry_run=False if wrapped function supports
                    # it, as to override any nested defaults saying otherwise
                    kwargs.pop('dry_run')
            else:
                dry_run = None
            retval = f(self, *args, **kwargs)
            if dry_run:
                transaction.set_rollback(True, dbalias)
            return retval
    return wrapper


def compile_ranges(int_list, min_range_size=2):
    """
    Compile ranges and singles from sequence of integers

    This is a helper for make_int_in_filter.

    Ranges will be (start, end) and inclusive, as for Django's range lookup and
    SQL's BETWEEN operator.
    """
    END = object()
    singles = []
    ranges = []
    range_min = None
    range_max = None

    ints = chain(sorted(set(int_list)), [END])
    first = next(ints)
    if first is END:
        return ranges, singles
    else:
        # initilize a range
        range_min = first
        range_max = first

    for i in ints:
        if i == range_max + 1:
            # extend current range
            range_max = i
        else:
            # finish current range
            if range_max - range_min + 1 >= min_range_size:
                ranges.append((range_min, range_max))
            else:
                for j in range(range_min, range_max + 1):
                    singles.append(j)
            # start new range
            range_min = i
            range_max = i
    return ranges, singles


def make_int_in_filter(lookup_name, integers):
    """
    Make a range lookup filter from a list of integers

    Use this for really long lists of mostly consequetive integers.  If the
    given list has only singles, or is empty, this will return the expected

        Q(<lookup_name>__in=integers)

    but for consequitive numbers __range lookups will be added as in

        Q(<lookup_name>__range=(start, end))

    The idea is that for most workloads this will result in a smaller SQL
    statement and maybe even a smaller query execution time.  In particular
    with sqlite2 we seem to run into some limit when using __in with a list of
    more than 250,000 integers, givinh us a "too many variables" error.
    """
    ranges, singles = compile_ranges(integers)
    q = Q(**{lookup_name + '__in': singles})
    for start, end in ranges:
        q = q | Q(**{lookup_name + '__range': (start, end)})
    return q


def save_import_diff(
    model,
    change_set=[],
    unchanged_count=None,
    new_count=None,
    new_pk_min=None,
    new_pk_max=None,
    missing_objs=[],
    path=None,
    dry_run=False,
):
    """
    Save change set to disk, called by Loader.load()
    """
    if path is None:
        path = settings.IMPORT_DIFF_DIR

    summary = (f'new: {new_count},  changed: {len(change_set)},  unchanged: '
               f'{unchanged_count},  missing: {len(missing_objs)}')
    print('Summary:', summary)
    now = datetime.now()

    if dry_run:
        opath = Path(path) / f'{model._meta.model_name}.dryrun.txt'
    else:
        # try to make a unique filename, but overwrite after last attempt
        def suffixes():
            yield '.txt'
            for i in range(1, 99):
                yield f'.{i}.txt'
        obase = Path(path) / f'{model._meta.model_name}.{now.date()}.txt'
        for suf in suffixes():
            opath = obase.with_suffix(suf)
            if not opath.exists():
                break

    def fmt(val):
        """ value formatter """
        if val is None:
            return '<None>'
        elif isinstance(val, str):
            return f'"{val}"'
        else:
            return val

    with opath.open('w') as ofile:
        ofile.write(f'=== {now} ===\nSummary:   {summary}\n')
        if new_pk_min is not None or new_pk_max is not None:
            ofile.write(
                f'New objects pk range (incl.): {new_pk_min} - {new_pk_max}\n'
            )
        if change_set:
            ofile.write('Changed fields as "pk id field: old -> new, ...":\n')
        for pk, record_id, items in change_set:
            items = [
                f'{field}: {fmt(old)}  ->  {fmt(new)}'
                for field, old, new in items
            ]
            ofile.write(f'{pk}\t{record_id}\t{items[0]}\n')
            for i in items[1:]:
                ofile.write(f'\t\t{i}\n')
        for pk, key in missing_objs:
            ofile.write(f'missing: {pk} / {key}\n')
    print(f'Diff saved to: {opath}')


class DefaultDict(dict):
    """
    A dict that can map unknown keys to a default value

    The default value is set via the default key, which by default is the
    string 'default'.  If there is a chance that the default key occurs in your
    data then one can get an instance with a different default key, e.g.:

    DEFAULT_KEY = object()
    d = SafeDict.with_default_key(DEFAULT_KEY)(...)

    If a default key-value pair is set then the default value is returned for
    any unknown key.  If no default key-value is set (or if a previous one was
    deleted) then the usual KeyError is raised when trying to access the
    dictionary with an unknown key.
    """
    default_key = 'default'

    @classmethod
    def with_default_key(cls, default_key):
        """
        Make a subclass with an alternative default key
        """
        if isinstance(default_key, str):
            name_suffix = default_key
        else:
            name_suffix = str(hash(default_key))

        name = cls.__name__ + '_' + name_suffix
        attrs = dict(default_key=default_key)
        return type(name, (cls, ), attrs)

    def __missing__(self, key):
        if self.default_key in self:
            return self[self.default_key]
        else:
            raise KeyError(key)
