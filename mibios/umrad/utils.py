from enum import Enum
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
from django.db import connection, router, transaction
from django.db.models import Q

# Workaround for weird pandas/xlrd=1.2/defusedxml combination runtime issue,
# we'll get an AttributeError: 'ElementTree' object has no attribute
# 'getiterator' inside xlrd when trying to pandas.read_excel().  See also
# https://stackoverflow.com/questions/64264563
import xlrd
if xlrd.__version__.startswith('1.2'):
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


class SkipRow(Exception):
    """
    Raised during Loader.load() to skip a row of input

    The condition may or may not be an error.  In case this indicates an error,
    it is not severe enough to abort loading the data, just to skip the one
    row.
    """
    def __init__(self, *args, log=True, **kwargs):
        self.log = log
        super().__init__(*args, **kwargs)


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
    for i in some_iterable:
        do_stuff()
        pp.inc()
    pp.finish()

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


class PrettyEnum(Enum):
    """ enum that prints just to its name  """
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class InputFileSpec:
    special_values = PrettyEnum('special_values',
                                ['IGNORE_COLUMN', 'CALC_VALUE', 'NO_HEADER'])
    IGNORE_COLUMN = special_values.IGNORE_COLUMN
    CALC_VALUE = special_values.CALC_VALUE
    NO_HEADER = special_values.NO_HEADER

    empty_values = []
    """
    A list of input-file-wide extra empty values.  For the purpose of loading
    the data these are used in addition to each field's empty_values attribute.
    """

    def __init__(self, *column_specs, has_header=None, extra=None):
        self._spec = column_specs or None

        # set by setup():
        self.model = None
        self.loader = None
        self.file = None
        self.has_header = has_header
        self.extra = extra or {}
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
        for specline in column_specs:
            # TODO: re-write as match statement?
            # NOTE: two-str-items format can be <col> <field> OR <field> <fun>
            # with no super easy way to distinguish them.  Use has_header init
            # arg to make it explicit if needed.
            if not isinstance(specline, tuple):
                # no-header-simple-format
                specline = (specline, )

            if self.has_header is None:
                # auto-detect header presence from first spec piece
                if len(specline) == 1:
                    self.has_header = False
                elif len(specline) == 2 and callable(specline[1]):
                    self.has_header = False
                else:
                    self.has_header = True

            if not self.has_header:
                specline = (self.NO_HEADER, *specline)

            colname, item2, *rest = specline

            if colname is self.CALC_VALUE:
                if not self.has_header:
                    # FIXME: this is unreachable, right?
                    col_index.append(self.CALC_VALUE)
            elif colname is self.NO_HEADER:
                # add 0-based numerical index for column name
                # these have to be counted here
                if cur_col_index is None:
                    cur_col_index = 0
                else:
                    cur_col_index += 1
                col_index.append(cur_col_index)

            get_field_err = None

            if item2 is self.IGNORE_COLUMN:
                continue
            elif isinstance(item2, str):
                field_name, _, fkattr = item2.partition('.')
                try:
                    field = self.model._meta.get_field(field_name)
                except FieldDoesNotExist as e:
                    # for now, assume it's supposed to be a prepfunc and not a
                    # field (but we keep the error to maybe show later)
                    get_field_err = e
                    prepfunc = field_name
                    key = field = field_name = None
                else:
                    prepfunc = None
                    key = item2  # TODO: the key is not needed anymore, right?
                    if fkattr:
                        self.fk_attrs[field_name] = fkattr
            elif callable(item2):
                # prepfunc given but no field
                key = field = field_name = None
                prepfunc = item2
            elif item2 is None:
                # explicit None key
                key = field = field_name = None
                prepfunc = None
            else:
                raise SpecError(f'Expected str or callable: {item2} in line '
                                f'{specline}')

            if rest:
                if prepfunc is not None:
                    raise SpecError(
                        f'unexpected third item {rest=} or second item '
                        f'({item2}) should have been key to a field: '
                        f'{get_field_err or "..."} | {specline}'
                    )
                if len(rest) > 1:
                    raise SpecError('too many items: {specline}')
                prepfunc = rest[0]
                if prepfunc is None:
                    pass
                elif isinstance(prepfunc, str):
                    prepfunc_name = prepfunc
                    # getattr gives us a bound method:
                    prepfunc = getattr(loader, prepfunc_name)
                    if not callable(prepfunc):
                        raise SpecError(
                            f'not the name of a {self.loader} method: '
                            f'{prepfunc_name} in spec {specline}'
                        )
                elif callable(prepfunc):
                    # Assume it's a function that takes the loader as
                    # 1st arg.  We get this when the previously
                    # declared method's identifier is passed directly
                    # in the spec's declaration.
                    prepfunc = partial(prepfunc, self.loader)
                else:
                    raise SpecError(f'expected a callable or manager method '
                                    f'name: {prepfunc}: {specline}')

            if key is None and colname is self.CALC_VALUE:
                raise SpecError('require key (field name) for for which'
                                'to calculate a value: {specline}')

            if key is None and prepfunc is None:
                # skip as there would be nothing to do
                continue

            if prepfunc is None and field and field.choices:
                prepfunc = self.loader.get_choice_value_prep_method(field)

            col_names.append(colname)
            keys.append(key)
            fields.append(field)
            field_names.append(field_name)
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

                value = value.strip()
                if value in self.empty_values:
                    value = None
                elif field is not None and value in field.empty_values:
                    value = None

            yield (field, fn, value)

    def row2dict(self, row_data):
        """ turn result of row_data() into a dict with field names as keys """
        return {field.name: val for field, _, val in row_data}


class ModelSpec(InputFileSpec):
    """
    Adapter to load prepared data

    To be used with a loader that provides a get_spec_column() method.
    """
    def __init__(self, rows):
        super().__init__(self)
        self._rows = rows

    def setup(self, loader):
        """
        Setup automatically all field declared in model.
        """
        super().setup(loader, column_specs=loader.get_spec_columns())

    def iterrows(self):
        return iter(self._rows)


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
    def __init__(self, *column_specs, sep='\t', **kwargs):
        super().__init__(*column_specs, **kwargs)
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
    Replacement for @atomic decorator for Manager methods (and others)

    Supports dry_run keyword arg and calls set_rollback appropriately and (if
    possible) coordinates the db alias in case we have multiple databases.
    This assumes that the decorated method has self.model available, as it is
    the case for managers, and that if write operations are used for other
    models then those must run on the same database connection.
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        try:
            dbalias = router.db_for_write(self.model)
        except Exception:
            using_kw = {}
            dbalias_args = tuple()
        else:
            using_kw = dict(using=dbalias)
            dbalias_args = (dbalias, )

        with transaction.atomic(**using_kw):
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
                transaction.set_rollback(True, *dbalias_args)
            return retval
    return wrapper


def compile_ranges(int_list, min_range_size=2, max_num_ranges=None):
    """
    Compile ranges and singles from sequence of integers

    This is a helper for make_int_in_filter.

    Ranges will be (start, end) and inclusive, as for Django's range lookup and
    SQL's BETWEEN operator.
    """
    if not min_range_size >= 2:
        raise ValueError('minimum range size must be 2 or larger')

    if max_num_ranges is not None and max_num_ranges < 1:
        # just being explicit, 0 would mess up the logic below
        raise ValueError('max_num_ranges must None or >= 1')

    ints = sorted(set(int_list))
    if not ints:
        return [], []

    END = object()
    singles = []
    ranges = []
    num_ranges = 0
    range_min = None
    range_max = None

    ints = chain(ints, [END])
    # initialize a range from first (smallest) element
    first = next(ints)
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
                num_ranges += 1
                if max_num_ranges and num_ranges >= max_num_ranges:
                    # just do singles from here on
                    singles.append(i)
                    break
            else:
                # range too small
                for j in range(range_min, range_max + 1):
                    singles.append(j)
            # start new range, assigns END before exiting
            range_min = i
            range_max = i

    singles.extend(ints)  # noop unless coming from break
    if singles and singles[-1] is END:
        # max_num_ranges was reached, rm end marker
        singles.pop()

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
    more than 250,000 integers, giving us a "too many variables" error.
    """
    if connection.vendor == 'sqlite':
        # avoiding "Expression tree is too large (maximum depth 1000)"
        max_num_ranges = 997
    else:
        max_num_ranges = None

    ranges, singles = compile_ranges(integers, max_num_ranges=max_num_ranges)
    q = Q(**{lookup_name + '__in': singles})
    for start, end in ranges:
        q = q | Q(**{lookup_name + '__range': (start, end)})
    return q


def save_import_log(
    model,
    skip_log,
    change_set=[],
    unchanged_count=None,
    new_count=None,
    new_pk_min=None,
    new_pk_max=None,
    missing_objs=[],
    delete_info=None,
    path=None,
    dry_run=False,
):
    """
    Save import log to disk, called by Loader.load()
    """
    if path is None:
        path = settings.IMPORT_LOG_DIR

    summary = (
        f'new: {new_count},  changed: {len(change_set)},  unchanged: '
        f'{unchanged_count},  missing: {len(missing_objs)},  '
        f'deleted: {delete_info}'
    )

    print('Summary:', summary)
    now = datetime.now()

    if dry_run:
        opath = Path(path) / f'{model._meta.model_name}.dryrun.log'
    else:
        # try to make a unique filename, but overwrite after last attempt
        def suffixes():
            yield '.txt'
            for i in range(1, 99):
                yield f'.{i}.txt'

        obase = f'{model._meta.model_name}.{now.date()}'
        if delete_info:
            obase += '.deleted'

        for suf in suffixes():
            opath = Path(path) / (obase + suf)
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
        for i in skip_log:
            # skip log items are dicts usually with lineno and messages keys
            out = ''
            if lineno := i.pop('lineno', None):
                out += f'Skipped data at {lineno}:'
            else:
                out += 'Skipped line(s):'
            msg = i.pop('message', '')
            items = ' '.join((f'{k}={v}' for k, v in i.items()))
            ofile.write(f'{out} {items}: {msg}\n')

    print(f'Import log saved to: {opath}')


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


class FKMap(dict):
    """
    Container to pre-load and index objects, helper for Loader.load()
    """
    def __init__(self, spec):
        """
        Create the FK mapping, usually called upfront in Loader.load()

        spec: The loader's fully initialized spec.

        This is a dict of dicts, mapping field names to maps from (tuples of)
        values to PKs.
        """
        self.to_pythons = {}  # any needed Field.to_python() functions
        for i in spec.fields:
            if i is None or not i.many_to_one:
                continue

            if i.name in spec.fk_attrs:
                # lookup field given by dot-notaton
                lookups = (spec.fk_attrs[i.name], )
            else:
                # use defaults
                lookups = i.related_model.get_accession_lookups()

            if lookups == ('pk', ) or lookups == ('id', ):
                # the values will be PKs
                continue

            self.to_pythons[i.name] = \
                [i.related_model._meta.get_field(j).to_python for j in lookups]

            f = spec.fkmap_filters.get(i.name, {})

            print(f'Retrieving {i.related_model._meta.verbose_name} data, '
                  f'{"filtered, " if f else ""}'
                  f'indexing by ({",".join(lookups)}) ...',
                  end='', flush=True)
            related_manager = getattr(i.related_model, 'loader', None) \
                or getattr(i.related_model, 'objects')
            if callable(f):
                self[i.name] = f()
            else:
                self[i.name] = {
                    tuple(a): pk for *a, pk
                    in related_manager.filter(**f).values_list(*lookups, 'pk')
                }
            print(f'[{len(self[i.name])} OK]')

    def get_pk(self, field, value):
        """
        Get the PK

        Returns None if no objects exists for the given value.

        Raises KeyError if we're not set up for given field (as expected in
        cases where the values are PKs already.)
        """
        self[field.name]  # check, raise KeyError as needed

        if not isinstance(value, tuple):
            value = (value, )  # fkmap keys are tuples

        try:
            value = tuple((
                fn(v) for fn, v
                in zip(self.to_pythons[field.name], value)
            ))
        except Exception as e:
            print(f'conversion to python values failed: {e}')
            raise e

        return self[field.name].get(value, None)
