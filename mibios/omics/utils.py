from contextlib import redirect_stdout
from datetime import datetime
from functools import wraps
from operator import methodcaller
import sys

from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def get_dataset_model():
    try:
        return django_apps.get_model(
            settings.OMICS_DATASET_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "OMICS_DATASET_MODEL must be of the form "
            "'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'OMICS_DATASET_MODEL refers to model '
            f'{settings.OMICS_DATASET_MODEL} that has not been installed'
        )


def get_sample_model():
    try:
        return django_apps.get_model(
            settings.OMICS_SAMPLE_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "OMICS_SAMPLE_MODEL must be of the form "
            "'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'OMICS_SAMPLE_MODEL refers to model '
            f'{settings.OMICS_SAMPLE_MODEL} that has not been installed'
        )


def get_fasta_sequence(file, offset, length, skip_header=True):
    """
    Retrieve sequence record from fasta formatted file with known offset

    parameters:
        file: file like object, opened for reading bytes
        offset: first byte of header
        length: length of data in bytes

    Returns the fasta record or sequence as bytes string.  The sequence part
    will be returned in a single line even if it was broken up into multiple
    line originally.
    """
    file.seek(offset)
    if skip_header:
        header = file.readline()
        if header[0] != ord(b'>'):
            raise RuntimeError('expected fasta header start ">" missing')
        length -= len(header)
        if length < 0:
            raise ValueError('header is longer than length')
    else:
        # not bothering with any checks here
        pass

    data = file.read(length).splitlines()
    if not skip_header:
        data.insert(1, b'\n')
    data = b''.join(data)
    return data


class parse_fastq:
    """
    Generate fastq records from something file-like.

    usage:
        for record in parse_fastq(open('my.fastq')):
            head = record['head']
            seq = record['seq']
            qual = record['qual']

    Records must be 4 lines long.  Returns a dict, trims off newlines and
    initial @ from header.
    """
    def __init__(self, file, skip_initial_trash=False):
        """
        Setup the fastq parser

        :param bool skip_initial_trash:
            If True, then any initial lines that don't parse as fastq will be
            skipped without raising an error.  This also assumes trat file is
            seekable.

        Parsing will not be reliable if seek() is called on the underying file
        t = descriptor after next() has been called already.
        """
        self.file = file
        if skip_initial_trash:
            pos = self.file.tell()
            while True:
                try:
                    self._make_record()
                except StopIteration:
                    # EOF
                    break
                except ValueError:
                    # advance one line and try again
                    self.file.seek(pos)
                    if self.file.readline():
                        pos = self.file.tell()
                    else:
                        # EOF
                        break
                else:
                    # there is a good record at pos
                    self.file.seek(pos)
                    break

    def __iter__(self):
        return self

    def __next__(self):
        return self._make_record()

    def _get_line(self):
        return next(self.file).rstrip('\n')

    def _make_record(self):
        """
        Build the record from buffer and next four lines

        Raises ValueError if checks for correct fastq format fail.
        """
        head = self._get_line()  # StopIteration here will bubble up
        if not head.startswith('@'):
            raise ValueError(f'expected @ at start of line: {head}')

        try:
            head = head.removeprefix('@')
            seq = self._get_line()
            if not seq:
                raise ValueError(f'sequence missing: {seq}')
            plus = self._get_line()
            if not plus.startswith('+'):
                raise ValueError(f'expected "+" at start of line: {plus}')
            qual = self._get_line()
            if len(qual) != len(seq):
                raise ValueError('sequence and quality differ in length')
            return dict(head=head, seq=seq, qual=qual)
        except StopIteration:
            # got a header but further lines missing
            raise ValueError('file ends with incomplete record')


def call_each(iterable, method_name, *args, **kwargs):
    """
    Call a method while iterating

    The method will be called and then the object is yield.  There are no means
    to retrieve the return values of the method calls.
    """
    callmeth = methodcaller(method_name, *args, **kwargs)
    for obj in iterable:
        callmeth(obj)
        yield obj


class Timestamper(redirect_stdout):
    """
    Context manager and proxy for e.g. sys.stdout to add timestamps to output

    Usage:
        with Timestamper():
            print('starting up...')
            do_work()  # writes to sys.stdout
            print('all done')
    """
    FULL_TIMESTAMP_CUTOFF = 3599  # in secs, when to switch to full time
    TS_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, file_obj=None, template=None, file_copy=None):
        """
        parameters:
          file_obj: the file-like to replace, defaults to sys.stdout
          template: Custom template may contain '{timestamp}' to print the time
                and date.  Defaults to print the timestamp in square brackets.
          file_copy:  Optional, path to hard-copy of output, to keep a
                persistent log.
        """
        self._file_obj = file_obj or sys.stdout
        self.cur_line = ''
        self.prev_timestamp = None
        self.indent = None
        if template is None:
            self.template = '[ {timestamp} ]  '
        else:
            self.template = template
        self._file_copy_path = file_copy
        self._file_copy = None
        # set up redirect_stdout with self as the new_target:
        super().__init__(self)

    # build file copy opening into redirect_stdout context management:
    def __enter__(self):
        if self._file_copy_path:
            self._file_copy = open(self._file_copy_path, 'a')
            self._last_nl = self._file_copy.tell()
        return super().__enter__()

    def __exit__(self, exctype, excinst, exctb):
        if self._file_copy:
            self._file_copy.close()
        super().__exit__(exctype, excinst, exctb)

    # Implement stdout proxying:
    def __getattr__(self, attr):
        return getattr(self._file_obj, attr)

    def _write(self, s):
        self._file_obj.write(s)
        if self._file_copy:
            if s == '\r':
                self._file_copy.seek(self._last_nl)
                self._file_copy.truncate()
            else:
                self._file_copy.write(s)
                if s.endswith('\n'):
                    self._last_nl = self._file_copy.tell()

    def write(self, s):
        if s.startswith('\r'):
            # process carriage return
            self._write('\r')
            self.cur_line = ''
            s = s.lstrip('\r')

        lines = s.splitlines(keepends=True)
        if not lines:
            return

        # compile prefix
        self.timestamp = datetime.now()
        if self.prev_timestamp is None:
            delta = None
        else:
            delta = self.timestamp - self.prev_timestamp
            if delta.total_seconds() > self.FULL_TIMESTAMP_CUTOFF:
                delta = None

        if delta is None:
            ts_str = self.timestamp.strftime(self.TS_FORMAT)
            self.full_ts_width = len(ts_str)
        else:
            seconds = delta.total_seconds()
            hours = int(seconds // 3600)
            seconds -= hours * 3600
            minutes = int(seconds // 60)
            seconds -= minutes * 60
            if hours:
                ts_str = f'+ {hours}:{minutes:02}:{seconds:02.0f}'
            elif minutes:
                ts_str = f'+ {minutes}:{seconds:02.0f}m'
            else:
                ts_str = f'+ {seconds:.0f}s'
                if ts_str == '+ 0s':
                    # zero seconds, leave blank
                    ts_str = ''
            ts_str = ts_str.rjust(self.full_ts_width)

        prefix = self.template.format(timestamp=ts_str)

        # rewrite current line w/new prefix and append first new line
        self.cur_line += lines.pop(0)
        self._write('\r')
        self._write(prefix + self.cur_line)

        # write any further lines, remember last line as current one
        while lines:
            self.cur_line = lines.pop(0)
            self._write(' ' * len(prefix) + self.cur_line)

        if self.cur_line.endswith('\n'):
            # that line is finished, so forget it
            self.cur_line = ''
            self.prev_timestamp = self.timestamp


def gentle_int(fn):
    """
    Function/method decorator to avoid traceback display when pressing Ctrl-C

    This will make the decorated function return None when a ^C is caught, so
    only use this on top-level functions/methods or otherwise be careful that
    the return value is not misinterpreted.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            # Pressing ctrl-c typically displays a ^C w/o newline, so the
            # message below starts with one.
            print(f'\n<returning from {fn} via keyboard interrupt>')
            return
    return wrapper
