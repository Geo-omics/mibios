from collections import Counter, defaultdict
from csv import DictReader, Sniffer
from inspect import signature
from io import TextIOBase, TextIOWrapper
import re
import sys

from django.core.exceptions import (FieldDoesNotExist, ObjectDoesNotExist,
                                    ValidationError)
from django.db import transaction, IntegrityError

from . import get_registry
from .dataset import PARSE_BLANK, UserDataError
from .models import ImportFile, NaturalKeyLookupError
from .utils import DeepRecord, getLogger


log = getLogger(__name__)
importlog = getLogger('dataimport')


class DryRunRollback(Exception):
    pass


class Loader():
    """
    Data importer
    """
    log = log
    dataset = None
    blanks = {None: ['']}
    parse_blank = []

    def __init__(self, data_name, sep=None, can_overwrite=True,
                 warn_on_error=False, strict_sample_id=False, dry_run=False,
                 user=None, erase_on_blank=False, no_new_records=False,
                 note=''):
        try:
            self.dataset = get_registry().datasets[data_name]
        except KeyError:
            self.model = get_registry().models[data_name]
        else:
            self.model = self.dataset.model

        model_name = self.model._meta.model_name
        if self.dataset:
            self.accr_map = {}
            for accr, *items in self.dataset.fields:
                try:
                    col, *extra = items
                except ValueError:
                    col = accr
                    extra = ()

                accr = model_name + '__' + accr
                self.accr_map[col.casefold()] = accr
                for i in extra:
                    if 'blanks' in i:
                        if accr not in self.blanks:
                            self.blanks[accr] = []
                        self.blanks[accr] += i['blanks']
                    if i == PARSE_BLANK:
                        self.parse_blank.append(accr)
        else:
            # set accessor map from model
            fields = self.model.get_fields(with_hidden=True)
            self.accr_map = {
                v.casefold(): model_name + '__' + n
                for v, n in zip(fields.verbose, fields.names)
            }
            if 'name' not in self.accr_map and hasattr(self.model, 'name'):
                self.accr_map['name'] = model_name + '__natural'

        self.warnings = []
        self.sep = sep
        self.new = defaultdict(list)
        self.added = defaultdict(lambda: defaultdict(list))
        self.changed = defaultdict(lambda: defaultdict(list))
        self.erased = defaultdict(lambda: defaultdict(list))
        self.count = 0
        self.line_key = {}
        self.fq_file_ids = set()
        self.can_overwrite = can_overwrite
        self.warn_on_error = warn_on_error
        self.strict_sample_id = strict_sample_id
        self.dry_run = dry_run
        self.user = user
        self.erase_on_blank = erase_on_blank
        self.no_new_records = no_new_records
        self.note = note
        self.file_record = None
        if dry_run:
            self.log = log
        else:
            self.log = importlog

        if self.dataset:
            self.blanks[None] += self.dataset.blanks

    def process_header(self):
        """
        Process the first row

        Helper for process_file()
        """
        self.ignored_columns = []  # columns that won't be processed
        for i in self.reader.fieldnames:
            if i.casefold() not in self.accr_map:
                self.ignored_columns.append(i)

        log.debug('accessor map:', self.accr_map)
        log.debug('ignored fields:', self.ignored_columns)

        if self.reader.fieldnames == self.ignored_columns:
            log.debug('input fields:', self.reader.fieldnames)
            raise UserDataError(
                'input file does not have any expected field/column names'
            )

    def pre_process_row(self, row):
        """
        Map file-fields to internal field names

        Remove fields not in spec, set blank fields to None
        Helper for process_row()
        """
        ret = {}
        for k, v in row.items():
            try:
                accessor = self.accr_map[k.casefold()]
            except KeyError:
                continue
            if self.is_blank(accessor, v):
                ret[accessor] = None
            else:
                ret[accessor] = v
        return ret

    def setup_reader(self, file):
        """
        Get the csv.DictReader all set up

        Helper for process_file()
        """
        if not isinstance(file, TextIOBase):
            # http uploaded files are binary
            file = TextIOWrapper(file)

        sniff_kwargs = {}
        reader_kwargs = {}
        if self.sep:
            sniff_kwargs['delimiters'] = self.sep
        try:
            dialect = Sniffer().sniff(file.read(5000), **sniff_kwargs)
        except Exception as e:
            log.debug('csv sniffer failed:', e)
            # trying fall-back (file might be too small)
            dialect = 'excel'  # set fall-bak default
            if self.sep:
                reader_kwargs['delimiter'] = self.sep
            else:
                reader_kwargs['delimiter'] = '\t'  # set fall-back default
        finally:
            file.seek(0)
            log.debug('sniffed:', vars(dialect))

        self.reader = DictReader(file, dialect=dialect, **reader_kwargs)
        self.sep = self.reader.reader.dialect.delimiter  # update if unset
        log.debug('delimiter:', '<tab>' if self.sep == '\t' else self.sep)
        log.debug('input fields:', self.reader.fieldnames)

    def process_file(self, file):
        """
        Load data from given file
        """
        log.debug('processing:', file, vars(file))
        self.linenum = 1
        self.last_warning = None
        row = None
        try:
            with transaction.atomic():
                self.file_record = ImportFile.create_from_file(
                    file=file,
                    note=self.note,
                )
                # Saving input file to storage: if the input file come from the
                # local filesystem, ImportFile.save() we need to seek(0) our
                # file handle.  Do uploaded files in memory do something else?
                file.seek(0)
                # Getting the DictReader set up must happen after saving to
                # disk as csv.reader takes some sort of control over the file
                # handle and disabling tell() and seek():
                self.setup_reader(file)
                self.process_header()

                for row in self.reader:
                    self.process_row(row)

                if self.dry_run:
                    raise DryRunRollback
        except Exception as e:
            if self.file_record is not None:
                self.file_record.file.delete(save=False)
                self.file_record = None
            if isinstance(e, DryRunRollback):
                pass
            elif isinstance(e, UserDataError):
                # FIXME: needs to be reported; and (when) does this happen?
                raise
            else:
                if row is None:
                    msg = 'error at file storage or opening stage'
                else:
                    msg = 'Failed processing line:\n{}'.format(row)
                raise RuntimeError(msg) from e

        return dict(
            count=self.count,
            new=self.new,
            added=self.added,
            changed=self.changed,
            erased=self.erased,
            ignored=self.ignored_columns,
            warnings=self.warnings,
            dry_run=self.dry_run,
            overwrite=self.can_overwrite,
            erase_on_blank=self.erase_on_blank,
            no_new_records=self.no_new_records,
            file_record=self.file_record,
        )

    def get_from_row(self, *keys):
        """
        Get a dict with specified keys based on row

        Helper method to update object templates
        """
        ret = {}
        for i in keys:
            if i in self.row:
                ret[i] = self.row[i]
        return ret

    def account(self, obj, is_new, from_row=None, is_primary_obj=False):
        """
        Accounting for object creation, change

        Enforce object overwrite as needed.
        Update state with object
        """
        model_name = obj._meta.model_name
        obj.add_change_record(
            file=self.file_record,
            line=self.linenum,
            user=self.user,
            comment=' '.join(sys.argv) if self.user is None else '',
        )
        need_to_save = False
        if is_new:
            self.new[model_name].append(obj)
            need_to_save = True
        elif from_row is not None:
            consistent, diffs = obj.compare(from_row)
            for k, v in from_row.items():
                apply_change = False

                if k in diffs['only_them']:
                    apply_change = True
                    self.added[model_name][obj].append(
                        (k, from_row.get(k))
                    )
                elif k in diffs['only_us']:
                    self.erased[model_name][obj].append(
                        (k, getattr(obj, k))
                    )
                    if self.erase_on_blank:
                        apply_change = True
                elif k in diffs['mismatch']:
                    self.changed[model_name][obj].append(
                        (k, getattr(obj, k), from_row.get(k))
                    )
                    if self.can_overwrite:
                        apply_change = True

                if apply_change:
                    need_to_save = True
                    setattr(obj, k, v)

        if need_to_save:
            obj.full_clean()

        if is_primary_obj:
            # check that each row corresponds to distinct record
            # (the primary record)
            if obj.natural in self.line_key:
                msg = (f'record {obj.natural} was already processed on line '
                       f'{self.line_key[obj.natural]}')
                raise UserDataError(msg)
            else:
                self.line_key[obj.natural] = self.linenum

        if need_to_save:
            obj.save()

    def is_blank(self, col_name, value):
        """
        Say if a value is "empty" or missing.

        An empty value is something like whitespace-only or Null or None
        or 'NA' or equal to a specified blank value etc.

        Values are assumed to be trimmed of whitespace already.
        """
        for i in self.blanks[None] + self.blanks.get(col_name, []):
            if isinstance(i, re.Pattern):
                if i.match(value):
                    return True
            elif value == i:
                return True
        if self.model.decode_blank(value) == '':
            return True
        return False

    def process_row(self, row):
        """
        Process a single input row

        This method does pre-processing and wraps the work into a transaction
        and handles some of the fallout of processing failure.  The actual work
        is delegated to process_fields().
        """
        self.linenum += 1
        self.row = self.pre_process_row(row)

        # rec: accumulates bits of processing before final assembly
        self.rec = {}
        # backup counters
        new_ = self.new.copy()
        added_ = self.added.copy()
        changed_ = self.changed.copy()
        erased_ = self.erased.copy()
        try:
            with transaction.atomic():
                self.process_fields()
        except (ValidationError, IntegrityError, UserDataError) as e:
            # Catch errors to be presented to the user;
            # some user errors in the data come up as IntegrityErrors, e.g.
            # violations of UNIQUE, IntegrityError should not be caught
            # inside an atomic() (cf. Django docs)
            if hasattr(e, 'message_dict'):
                # ValidationError from full_clean()
                # TODO: format msg dict?
                msg = str(e.message_dict)
            elif hasattr(e, 'messages'):
                # other Validation Error (e.g. to_python())
                msg = ' '.join(e.messages)
            else:
                msg = str(e)

            if not self.warn_on_error:
                # re-raise with row info added
                msg = 'at line {}: {}, current row:\n{}' \
                      ''.format(self.linenum, msg, self.row)
                raise type(e)(msg) from e

            err_name = type(e).__name__
            self.warnings.append(
                'skipping row: at line {}: {} ({})'
                ''.format(self.linenum, msg, err_name)
            )

            # manage repeated warnings
            if self.last_warning is None:
                repeats = 0
            else:
                last_err, last_msg, last_line, repeats = self.last_warning
                if msg == last_msg and last_line + 1 == self.linenum:
                    # same warning as for previous line
                    self.warnings.pop(-1)  # rm repeated warning
                    repeats += 1
                    if repeats > 1:
                        # rm repeater line
                        self.warnings.pop(-1)
                    self.warnings.append(
                        '    (and for next {} lines)'.format(repeats)
                    )
                else:
                    # warning was new
                    repeats = 0

            self.last_warning = (err_name, msg, self.linenum, repeats)

            # reset stats:
            self.new = new_
            self.added = added_
            self.changed = changed_
            self.erased = erased_
        except Exception as e:
            msg = 'at line {}: {}, current row:\n{}' \
                  ''.format(self.linenum, e, self.row)
            raise type(e)(msg) from e

        self.count += 1

    @classmethod
    def load_file(cls, file, data_name=None, **kwargs):
        loader = cls(data_name, **kwargs)
        return loader.process_file(file)

    def parse_value(self, accessor, value):
        """
        Delegate to specified Dataset.parse_FOO method
        """
        # rm model prefix from accsr to form method name
        pref, _, a = accessor.partition('__')
        parse_fun = getattr(self.dataset, 'parse_' + a, None)

        if parse_fun is None:
            ret = value
        else:
            args = [value]
            if len(signature(parse_fun).parameters) == 2:
                args.append(self.rec)

            try:
                ret = parse_fun(*args)
            except Exception as e:
                # assume parse_fun is error-free and blame user
                for i, j in self.accr_map.items():
                    if j == accessor:
                        col = i
                        break
                else:
                    col = '??'
                raise UserDataError(
                    'Failed parsing value "{}" in column {}: {}:{}'
                    ''.format(value, col, type(e).__name__, e)
                ) from e

        if isinstance(ret, dict):
            # put prefix back
            ret = {pref + '__' + k: v for k, v in ret.items()}

        return ret

    def get_model(self, accessor):
        """
        helper to get model class from accessor
        """
        name = accessor[0]
        m = get_registry().models[name]  # may raise KeyError

        for i in accessor[1:]:
            try:
                m = m._meta.get_field(i).related_model
            except (FieldDoesNotExist, AttributeError) as e:
                raise LookupError from e
            if m is None:
                raise LookupError
        return m

    def process_fields(self):
        """
        Process a row column by column

        Column representing non-foreign key fields are processed first and used
        to get/create their objects, root objects are created last
        """
        self.rec = DeepRecord()
        for k, v in self.row.items():
            if v is not None or k in self.parse_blank:
                v = self.parse_value(k, v)

                if self.dataset and v is self.dataset.IGNORE_THIS_FIELD:
                    continue

            if isinstance(v, dict):
                self.rec.update(**v)
            else:
                self.rec[k] = v

        msg = 'line {}: record: {}'.format(self.linenum, self.rec)
        if self.dry_run:
            log.debug(msg)
        else:
            importlog.info(msg)

        for k, v in self.rec.items(leaves_first=True):
            model, id_arg, obj, new = [None] * 4
            _k, _v, data = [None] * 3
            try:
                try:
                    # try as model
                    model = self.get_model(k)
                except LookupError:
                    # assume a field
                    continue

                if isinstance(v, dict):
                    data = v.copy()
                    id_arg = {}
                elif isinstance(v, model):
                    # value was instantiated by parse_value()
                    # nothing left to do, skip accounting
                    continue
                elif v:
                    id_arg = dict(natural=v)
                    data = {}
                elif v is None:
                    # TODO: ?not get correct blank value for field?
                    continue
                else:
                    raise RuntimeError(
                        'oops here: data: {}\nk:{}\nv:{}\nstate:{}'
                        ''.format(data, k, v, self.rec)
                    )

                # separate identifiers from other fields
                for i in ['natural', 'id', 'name']:
                    if i in data:
                        id_arg[i] = data.pop(i)

                # separate many_to_many fields from data
                m2ms = {}
                for _k, _v in data.items():
                    try:
                        field = model._meta.get_field(_k)
                    except FieldDoesNotExist:
                        continue
                    if field.many_to_many:
                        m2ms[_k] = _v

                for i in m2ms:
                    del data[i]
                m2ms = {
                    _k: _v for _k, _v in m2ms.items()
                    # filter out Nones
                    if _v is not None
                }

                # convert str field values to correct python type:
                # (a bit) link Field.to_python()
                # We do this here outside of the usual Model.full_clean() to
                # get the correct values to compare them with existing objects
                # in account() below.
                data1 = {}
                field = None
                for _k, _v in data.items():
                    field = model._meta.get_field(_k)
                    if _v is None:
                        # ensure correct blank values
                        if field.null:
                            data1[_k] = None
                        elif field.blank:
                            data1[_k] = ''
                        else:
                            # rm the field, will get default value for new objs
                            # TODO: issue a warning
                            continue
                    elif isinstance(_v, str):
                        # may raise ValidationError
                        data1[_k] = field.to_python(_v)
                    else:
                        # non-str values are coerced already
                        data1[_k] = _v
                data = data1
                del data1, field

                # if we don't have unique ids, use the "data" instead
                if not id_arg:
                    id_arg = data
                    data = {}

                try:
                    obj = model.objects.get(**id_arg)
                except model.DoesNotExist as e:
                    if self.no_new_records:
                        raise UserDataError('record not found') from e
                    # id_arg was used as lookup in get() above but used now for
                    # the model constructor, this works as long as the keys are
                    # limited to field or property names
                    try:
                        obj = model(**id_arg, **data)
                    except ObjectDoesNotExist as e:
                        # rel obj within natural key missing
                        raise UserDataError(f'record not found: {e}') from e
                    new = True
                except model.MultipleObjectsReturned as e:
                    # id_arg under-specifies
                    msg = '{} is not specific enough for {}' \
                          ''.format(id_arg, model._meta.model_name)
                    raise UserDataError(msg) from e
                except NaturalKeyLookupError as e:
                    raise UserDataError(e) from e
                except ValueError as e:
                    # happens for value of wrong type, e.g. non-number in an id
                    # field so int() fails, and who knows, maybe other reasons,
                    # anyways, let's blame the user for uploading bad data.
                    raise UserDataError(
                        'Possibly bad value / type not matching the field: {}:'
                        '{}'.format(type(e).__name__, e)
                    ) from e
                else:
                    new = False

                # is this the row's primary record?
                is_primary_obj = \
                    len(k) == 1 and k[0] == self.model._meta.model_name

                self.account(obj, new, data, is_primary_obj=is_primary_obj)

                for _k, _v in m2ms.items():
                    getattr(obj, _k).add(_v)

                # replace with real object
                self.rec[k] = obj

            except (IntegrityError, UserDataError, ValidationError):
                raise
            except Exception as e:
                raise RuntimeError(
                    'k={} v={} model={} id_arg={}\ndata={}\nrow record=\n{}'
                    ''.format(k, v, model, id_arg, data,
                              self.rec.pretty(indent=(3, 2)))
                ) from e
