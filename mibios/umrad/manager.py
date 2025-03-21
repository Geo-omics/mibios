from collections import defaultdict
from decimal import Decimal
from functools import partial
from inspect import trace
from itertools import islice
from logging import getLogger
from operator import attrgetter, length_hint
from time import sleep

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import connection, transaction
from django.utils.module_loading import import_string

from mibios.managers import BaseManager as MibiosBaseManager
from mibios.query import QuerySet as MibiosQuerySet

from .utils import (CSV_Spec, ProgressPrinter, atomic_dry, InputFileError,
                    get_last_timer, make_int_in_filter, save_import_log,
                    FKMap, SkipRow)


log = getLogger(__name__)


class BulkCreateWrapperMixin:
    """
    Mixin providing the bulk_create wrapper

    Intended to be mixed into both QuerySet and Manager classes.  Inside
    QuerySet it is used to overwrite the bulk_create method.  <add some more
    here...>
    """
    @staticmethod
    def bulk_create_wrapper(wrappee_bc):
        """
        A wrapper to add batching and progress metering to bulk_create()

        :param bound method wrappee_bc: A create_bulk method bound to a manager

        This static wrapper method allows us to still (1) override the usual
        Manager.bulk_create() but also (2) offer a wrapper to be used for cases
        in which we are unable or don't want to override the model's manager,
        e.g. the implicit through model of a many-to-many relation.
        """
        def bulk_create(objs, batch_size=None, ignore_conflicts=False,
                        progress_text=None, progress=True):
            """
            Value-added bulk_create

            Does not return the object list like super().bulk_create() does.
            """
            if not progress:
                # behave like original method, sans return value
                wrappee_bc(
                    objs,
                    ignore_conflicts=ignore_conflicts,
                    batch_size=batch_size,
                )
                return

            if batch_size is None:
                if connection.vendor == 'sqlite':
                    # sqlite has a variable-per-query limit of 999; it's not
                    # clear how one runs against that; it seems that multiple
                    # smaller INSERTs are being used automatically.  So, until
                    # further notice, only have a default batch size for
                    # progress metering here.
                    batch_size = 999
                else:
                    # allow progress update every few seconds
                    batch_size = 25000

            # get model name from manager (wrappee_bc.__self__ is the manager)
            model_name = wrappee_bc.__self__.model._meta.verbose_name

            if progress_text is None:
                if model_name.endswith(' relationship'):
                    # m2m through model
                    rec_type = 'link'
                else:
                    rec_type = 'record'
                progress_text = f'{model_name} {rec_type}s created'

            objs = iter(objs)
            pp = ProgressPrinter(
                progress_text,
                length=length_hint(objs) or None,
            )

            while True:
                batch = list(islice(objs, batch_size))
                if not batch:
                    break
                try:
                    wrappee_bc(
                        batch,
                        ignore_conflicts=ignore_conflicts,
                    )
                except Exception as e:
                    print(f'ERROR saving to {model_name or "?"}: batch 1st: '
                          f'{vars(batch[0])=}')
                    raise RuntimeError('error saving batch', batch[:9]) from e
                pp.inc(len(batch))

            pp.finish()

        return bulk_create

    @staticmethod
    def bulk_update_wrapper(wrappee_bu):
        """
        A wrapper to add batching and progress metering to bulk_update()

        :param bound method wrappee_bu: bulk_update method bound to a manager

        This static wrapper method allows us to still (1) override the usual
        Manager.bulk_update() but also (2) offer a wrapper to be used for cases
        in which we are unable or don't want to override the model's manager,
        e.g. the implicit through model of a many-to-many relation.
        """
        def bulk_update(objs, fields, batch_size=None, progress_text=None,
                        progress=True):
            """
            Value-added bulk_update
            """
            if not progress:
                # behave like original method
                wrappee_bu(objs, fields, batch_size=batch_size)
                return

            if batch_size is None:
                # sqlite has a variable-per-query limit of 999; it's not clear
                # how one runs against that; it seems that multiple smaller
                # INSERTs are being used automatically.  So, until further
                # notice, only have a default batch size for progress metering
                # here.
                batch_size = 999

            # get model name from manager (wrappee_bu.__self__ is the manager)
            model_name = wrappee_bu.__self__.model._meta.verbose_name

            if progress_text is None:
                progress_text = f'{model_name} records updated'

            objs = iter(objs)
            pp = ProgressPrinter(
                progress_text,
                length=length_hint(objs) or None,
            )

            while True:
                batch = list(islice(objs, batch_size))
                if not batch:
                    break
                try:
                    wrappee_bu(batch, fields)
                except Exception as e:
                    print(f'ERROR updating {model_name or "?"}: batch 1st: '
                          f'{vars(batch[0])=}')
                    print(f'{len(batch)=} {batch[:10]=}')
                    print(f'{batch[-10:]=}')
                    raise RuntimeError('error updating batch') from e
                pp.inc(len(batch))

            pp.finish()

        return bulk_update


class QuerySet(BulkCreateWrapperMixin, MibiosQuerySet):
    def search(self, query_term, field_name=None, lookup=None):
        """
        implement search from search field

        For models for which get_accession_field_single raises an exception,
        this method should be overwritten.
        """
        if field_name is None:
            uid_field = self.model.get_search_field()
        else:
            uid_field = self.model._meta.get_field(field_name)

        if lookup is None:
            if uid_field.get_internal_type() in ('CharField', 'TextField'):
                lookup = 'icontains'
            else:
                lookup = 'exact'

        if prefix := getattr(uid_field, 'prefix', ''):
            # rm prefix from query for accession fields
            if query_term.casefold().startswith(prefix.casefold()):
                query_term = query_term[len(prefix):]

        kw = {uid_field.name + '__' + lookup: query_term}
        try:
            return self.filter(**kw)
        except ValueError:
            # wrong type, e.g. searching for "text" on a numeric field
            return self.none()

    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False,
                    progress_text=None, progress=True):
        """
        Value-added bulk_create with batching and progress metering

        Does not return the object list like Django's QuerySet.bulk_create()
        does.
        """
        wrapped_bc = self.bulk_create_wrapper(super().bulk_create)
        wrapped_bc(
            objs=objs,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
            progress_text=progress_text,
            progress=progress,
        )

    def bulk_update(self, objs, fields, batch_size=None, progress_text=None,
                    progress=True):
        """
        Value-added bulk_update with batching and progress metering
        """
        wrapped_bu = self.bulk_update_wrapper(super().bulk_update)
        wrapped_bu(
            objs,
            fields,
            batch_size=batch_size,
            progress_text=progress_text,
            progress=progress,
        )

    def pkmap_vals(self, field_name, id_list=None):
        """
        Get map (as list of pairs) from (unique) values to PKs.

        Returns a list of 2-tuples.

        To get these as a dict use pkmap().  Similar to in_bulk() but arguments
        are reversed.  Argument field_name has no default as 'pk' does not make
        sense here.  Faster than in_bulk as targets are not objects just the
        primary keys.

        The values for given field should be unique within the queryset but
        this is not checked.  Likewise it is not an error if values of ids are
        not actually found in the database.
        """

        if id_list is None:
            return self.values_list(field_name, 'pk')

        # PG on docker flames out (invalid memory alloc request size) at batch
        # size about >8 million (when used as in UniRefMixin), sqlite has a
        # limit of 250000
        BATCH_SIZE = 1000000
        if connection.vendor == 'sqlite':
            BATCH_SIZE = 250000

        res = []
        for i in range(0, len(id_list) - 1, BATCH_SIZE):
            f = {f'{field_name}__in': id_list[i:i + BATCH_SIZE]}
            part = self.filter(**f).values_list(field_name, 'pk')
            res += part
        return res

    def pkmap(self, field_name, id_list=None):
        """
        Get map from (unique) values (accessions etc.) to PKs

        Like pkmap_vals() but returns the mapping properly as dict.
        """
        return dict(self.pkmap_vals(field_name, id_list))


class BaseLoader(MibiosBaseManager):
    """
    A manager providing functionality to load data from file
    """
    empty_values = []

    _DEFAULT_LOAD_KWARGS = dict(
        sep=None,
        skip_on_error=0,
        delete=False,
        update=False,
        strict_update=False,
        bulk=False,
        validate=False,
        diff_stats=False,
        no_input=False,
    )
    # Inheriting classes can set default kwargs for load() here.  In __init__()
    # anything missing is set from _DEFAULT_LOAD_KWARGS above, which inheriting
    # classes should not change (assuming they want to use our load().
    default_load_kwargs = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._DEFAULT_LOAD_KWARGS.items():
            self.default_load_kwargs.setdefault(k, v)

    def get_file(self, *args, **kwargs):
        """
        Return pathlib.Path to input data file
        """
        raise NotImplementedError('must be implemented in inheriting model')

    spec = None
    """
    A list of field names or list of tuples matching file headers to field
    names.  Use list of tuples if import file has a header.  Otherwise use the
    simple list, listing the fields in the right order.
    """

    def load(self, spec=None, start=0, limit=None, dry_run=False,
             parse_into=None, file=None, template={}, **kwargs):
        """
        Load data from file

        :param int start:
            Skip this many records at beginning of file (not counting header)
        :param int limit:
            Stop after reading this many records.  If None (the default), then
            load all data.
        :param list parse_into:
            If an empty list is passed, then we run in "parse only" mode.  The
            parsed data is appended to the provided list.
        :param int skip_on_error:
            Skip this many lines with errors if >0.  Certain Exceptions raised
            while processing a line of the input file cause the line to be
            skipped and an error message to be printed.  If set to the bool
            True, then all lines with errors will be skipped.
        :param bool validate:
            Run full_clean() on new model instances.  By default we skip this
            step as it hurts performance.  Turn this on if the DB raises in
            issue as this way the error message will point to the offending
            line.
        :param bool delete:
            Set to True for delete mode.  Defaults to False.  In delete mode
            the input file is read as usual and records already in the database
            but which do not occur in the input file are deleted.  Nothing else
            changes, no updates or record creations.
        :param bool no_input:
            If True, then this NOT prompt the user to confirm deleting data
            (when in delete mode).  Defaults to False.

        """
        # Most work is delegated to _load_rows(), here in the body of load() we
        # just do some setup and clean-up stuff.

        # set default kwargs if any are missing
        for k, v in self.default_load_kwargs.items():
            kwargs.setdefault(k, v)

        if kwargs['delete'] and not kwargs['update']:
            raise ValueError(
                'Incompatible options: delete mode requires the update option'
            )

        # setup spec
        setup_kw = dict(spec=spec)
        if kwargs['sep'] is not None:
            setup_kw['sep'] = kwargs['sep']
        if file is not None:
            setup_kw['file'] = file
        self.setup_spec(**setup_kw)

        for fn in self.spec.pre_load_hook:
            fn()

        row_it = self.spec.iterrows()
        self.current_lineno = None

        if start > 0 or limit is not None:
            if limit is None:
                stop = None
            else:
                stop = start + limit
            row_it = islice(row_it, start, stop)

        if parse_into is not None:
            self._parse_rows(row_it, parse_into=parse_into)
            return

        if kwargs['skip_on_error'] is True:
            # If True, let's skip all errors, so set to negative value here, if
            # decremented later it will never reach zero
            kwargs['skip_on_error'] = -1

        try:
            diff, skip_log = self._load_rows(
                row_it,
                template=template,
                dry_run=dry_run,
                first_lineno=start + 2 if self.spec.has_header else 1,
                **kwargs
            )
        except Exception as e:
            # cleanup progress printing (if any)
            try:
                get_last_timer().cancel()
                print()
            except Exception:
                pass

            if isinstance(e, InputFileError):
                print(f'Error in input file, line {self.current_lineno}: {e}')
                fi = trace(context=2)[-1]  # get FrameInfo obj
                print(f'Origin: {fi.filename}:{fi.function}():{fi.lineno}')
                print('Code context:')
                print(*fi.code_context, sep='')
                print('Local variables:')
                for k, v in reversed(fi.frame.f_locals.items()):
                    # Let's be careful here, only print simple variables and
                    # lengths for collections, anything else may ask for
                    # trouble
                    if isinstance(v, (int, float, type(None), bool)):
                        print(f'  {k}: {v}')
                    if isinstance(v, str):
                        print(f'  {k}: "{v}"')
                    elif hasattr(v, '__len__'):
                        print(f'  {k}: {type(v)} of length {len(v)}')
                    else:
                        pass
                fi.frame.clear()
                return
            else:
                raise

        # _load_rows returned
        something_changed = any((
            diff.get(i, None)
            for i
            in ('delete_info', 'new_count', 'change_set', 'missing_objs')
        ))

        if something_changed or skip_log:
            try:
                log_dir = settings.IMPORT_LOG_DIR
            except AttributeError:
                log_dir = ''

            if log_dir:
                save_import_log(self.model, skip_log, **diff, dry_run=dry_run)
            else:
                print('WARNING: IMPORT_LOG_DIR not configured - not '
                      'saving import log')
        else:
            print('nothing to log / diff: no changes recorded')

    def setup_spec(self, spec=None, **kwargs):
        if spec is None:
            if self.spec is None:
                raise NotImplementedError(f'{self}: loader spec not set')
        else:
            self.spec = spec
        self.spec.setup(loader=self, **kwargs)

    @atomic_dry
    def _load_rows(
        self,
        rows,
        sep='\t',
        template={},
        skip_on_error=0,
        validate=False,
        delete=False,
        update=False,
        strict_update=False,
        bulk=True,
        no_input=False,
        first_lineno=None,
        diff_stats=False,
        blank_invalid=True,
    ):
        """
        Do the data loading, called by load()

        Returns a tuple (diffs dict, skip log list).
        """
        fields = [i for i in self.spec.fields if i is not None]
        model_name = self.model._meta.model_name

        if update:
            print(f'Retrieving {model_name} records for update mode... ',
                  end='', flush=True)
            obj_pool = self._get_objects_for_update(template)
            print(f'[{len(obj_pool)} OK]')

        # loading info to process FKs
        fkmap = FKMap(self.spec)

        pp = ProgressPrinter(f'{model_name} rows read from file')
        new_objs = []  # will be created
        upd_objs = []  # will be updated
        m2m_data = {}
        missing_fks = defaultdict(set)
        skipped_log = []
        blanked_values = []
        skipped_log_not_logged = 0
        fk_skip_count = 0
        field = pk = value = obj = m2m = None

        for lineno in self.iterate_rows(pp(rows), start=first_lineno):
            obj = None
            obj_is_new = None
            m2m = {}
            for field, fn, value in self.current_row_data:

                if callable(fn):
                    try:
                        value = fn(value, obj)
                    except SkipRow as e:
                        if e.log:
                            skipped_log.append(dict(
                                lineno=lineno,
                                field=field,
                                value=value,
                                message=' / '.join((str(i) for i in e.args)),
                            ))
                        else:
                            skipped_log_not_logged += 1
                        break  # skips line / avoids for-else block
                    except Exception as e:
                        if skip_on_error and isinstance(e, InputFileError):
                            # skip line on expected errors
                            skipped_log.append(dict(
                                lineno=lineno,
                                field=field,
                                value=value,
                                row=self.current_row,
                                type=type(e).__name__,
                                message=' / '.join((str(i) for i in e.args)),
                            ))
                            break  # skips line / avoids for-else block
                        else:
                            print(f'\nERROR at line {lineno} / field {field}: '
                                  f'value was "{value}" -- {e}')
                            raise

                if value is self.spec.IGNORE_COLUMN or field is None:
                    continue  # next column

                # obj is always None for first field
                if obj is None and update:
                    # Try locating obj in obj_pool; first field MUST identify
                    # the object, the value's type must match that of the obj
                    # pool's keys
                    if value is None:
                        skipped_log.append(dict(
                            lineno=lineno,
                            field=field,
                            message='no record ID in row\'s first item',
                        ))
                        break  # skips line

                    if field.many_to_one and field.name in fkmap:
                        pool_key = fkmap.get_pk(field, value)
                    else:
                        pool_key = field.to_python(value)

                    try:
                        obj = obj_pool[pool_key]
                    except KeyError:
                        if strict_update:
                            raise InputFileError(
                                f'strict update violation at line {lineno}: '
                                f'record with ID value(s) "{pool_key}" does '
                                f'not exist in DB'
                            )
                        else:
                            # not found, obj remains None, create new one below
                            pass
                    else:
                        # ensure obj is only updated once
                        if obj is None:
                            raise InputFileError(
                                f'duplicate key value: {value} at line '
                                f'{lineno}'
                            )
                        obj_pool[pool_key] = None
                        obj_is_new = False
                        # the value is already set, go to next field please
                        continue

                if obj is None:
                    obj = self.model(**template)
                    obj_is_new = True

                if field.many_to_many:
                    # the "value" here is a list of tuples of through model
                    # constructor parameters without the local object(s)
                    # pk/id.  For normal, automatically generated, through
                    # models this is just the remote object's accession (which
                    # will later be replaced by the pk/id.)  If value comes in
                    # as a str, this list-of-tuples is generated below.  If it
                    # is neither None nor a str, then we assume that a
                    # pre-processing method has taken care of everything.  For
                    # through models with additional data the parameters must
                    # be in the correct order.
                    if value is None:
                        value = []  # blank / empty
                    elif isinstance(value, str):
                        value = [(i, ) for i in self.split_m2m_value(value)]
                    m2m[field.name] = value

                elif value is None:
                    # blank / empty value -- any non-m2m field type
                    setattr(obj, field.name, field.get_default())

                elif field.many_to_one:
                    try:
                        pk = fkmap.get_pk(field, value)
                    except KeyError:
                        # value is PK
                        if isinstance(value, int):
                            pk = value
                        else:
                            pk = None

                    if pk is None:
                        # FK target object does not exist
                        missing_fks[field.name].add(value)
                        if field.null:
                            pass
                        else:
                            fk_skip_count += 1
                            break  # skip this obj / skips for-else block
                    else:
                        setattr(obj, field.attname, pk)
                else:
                    # regular field with value
                    # TODO: find out why leaving '' in for int fields fails
                    # ValueError @ django/db/models/fields/__init__.py:1825
                    setattr(obj, field.name, value)
            else:  # the for else / row not skipped / keep obj
                if validate:
                    try:
                        blanked_log = self._validate(obj, blank_invalid)
                    except ValidationError as e:
                        print(f'\nERROR on line {lineno}: '
                              f'{"(new)" if obj_is_new else "(update)"} {e}'
                              f'offending row:\n{self.current_row}')
                        print(f'{vars(obj)=}')
                        if skip_on_error:
                            print(f'-- skipped row,  line: {lineno} --')
                            skip_on_error -= 1
                            continue  # skips the line
                        raise
                    else:
                        if blanked_log:
                            for i in blanked_log:
                                i['lineno'] = lineno
                            blanked_values.extend(blanked_log)

                if obj_is_new:
                    new_objs.append(obj)
                else:
                    upd_objs.append(obj)

                if m2m:
                    m2m_data[obj.get_accessions(ensure_type=True)] = m2m

        del fkmap, field, value, pk, obj, m2m

        if update or delete:
            missing_objs = [(j.pk, i) for i, j in obj_pool.items() if j]
            del obj_pool
            if not delete:
                if missing_objs:
                    print(f'WARNING: {len(missing_objs)} existing {model_name}'
                          f' records not updated (missing from input data)')
                if not diff_stats:
                    del missing_objs

        sleep(0.2)  # let the progress meter finish before printing warnings

        for fname, bad_ids in missing_fks.items():
            bad_id_list = [
                [str(j) for j in i] if isinstance(i, tuple) else str(i)
                for i in islice(bad_ids, 5)
            ]
            print(f'WARNING: found {len(bad_ids)} distinct unknown {fname} '
                  'IDs:', ' '.join(bad_id_list),
                  '...' if len(bad_ids) > 5 else '')
        if fk_skip_count:
            print(f'WARNING: skipped {fk_skip_count} rows due to unknown but '
                  f'non-null FK IDs')
        if skip_total := skipped_log_not_logged + len(skipped_log):
            print(f'Skipped {skip_total} rows (see import log for details)')
            if skipped_log_not_logged:
                skipped_log.append(dict(message=f'skipped (but no details'
                                   f' logged): {skipped_log_not_logged}'))
            del skip_total
        del skipped_log_not_logged
        if blanked_values:
            print(f'WARNING: {len(blanked_values)} invalid values got blanked/'
                  f'nulled (see import log for details)')
        skipped_log.extend(blanked_values)
        del blanked_values

        if not new_objs and not upd_objs:
            # Assume the empty file (or what else?) is an error.  In
            # particular, an empty input file can not be used to delete all
            # data.  Accordingly, all data changes happen further down in this
            # method.
            print('WARNING: nothing saved, empty file or all got skipped')
            return {}, skipped_log

        if missing_fks:
            del fname, bad_ids, bad_id_list
        del missing_fks, fk_skip_count

        update_fields = [i.name for i in fields if not i.many_to_many]

        diff = {}
        if diff_stats:
            if update or delete:
                diff['missing_objs'] = missing_objs

            if not delete:
                diff['change_set'] = []
                diff['unchanged_count'] = 0
                diff['new_count'] = len(new_objs)

                # retrieve old objects again
                upd_q = make_int_in_filter('pk', [i.pk for i in upd_objs])
                old_objs = self.filter(upd_q)
                upd_objs_by_pk = {i.pk: i for i in upd_objs}  # for fast access
                old_value, upd_value, items = None, None, None
                for i in old_objs:
                    # compile differences
                    items = []
                    for j in update_fields:
                        old_value = getattr(i, j)
                        upd_value = getattr(upd_objs_by_pk[i.pk], j)
                        if isinstance(old_value, Decimal) and isinstance(upd_value, Decimal):  # noqa:E501
                            # since Decimal('1') == Decimal('1.0'), but we want
                            # to tell the difference
                            is_equal = lambda a, b: not a.compare_total(b)  # noqa: E731,E501
                        else:
                            is_equal = lambda a, b: a == b  # noqa: E731
                        if not is_equal(old_value, upd_value):
                            items.append((j, old_value, upd_value))
                    if items:
                        diff['change_set'].append(
                            (i.pk, getattr(i, fields[0].name), items)
                        )
                    else:
                        diff['unchanged_count'] += 1
                del upd_q, old_objs, old_value, upd_value, items

        if delete:
            del_info = self.model._base_manager \
                           .filter(pk__in=[pk for pk, _ in missing_objs]) \
                           .delete()
            if not no_input:
                input(
                    f'Deleted objects info:\n'
                    f'{del_info}\n'
                    f'Press ^C to abort the transaction.\n'
                    f'Press <Enter> to commit to the deletion.\n'
                )
            # delete mode: only deletes data, so no
            # update or creation just return here.
            if diff_stats:
                if del_info != (0, {}):
                    diff['delete_info'] = del_info

        if upd_objs:
            if bulk:
                self.fast_bulk_update(upd_objs, fields=update_fields)
            else:
                pp = ProgressPrinter(f'{model_name} records updated')
                for i in pp(upd_objs):
                    try:
                        i.save(update_fields=update_fields)
                    except Exception as e:
                        print(f'exception {e} while saving {i}:\n{vars(i)=}')
                        raise
        del upd_objs, update_fields

        if new_objs:
            if bulk:
                self.bulk_create(new_objs)
            else:
                pp = ProgressPrinter(f'new {model_name} records saved')
                for i in pp(new_objs):
                    try:
                        i.save()
                    except Exception as e:
                        print(f'ERROR: {type(e)}: {e} while saving object: {i}'
                              f'\n{vars(i)=}')
                        raise
            if diff_stats:
                diff['new_pk_min'] = new_objs[0].pk
                diff['new_pk_max'] = new_objs[-1].pk
        if diff_stats:
            diff['new_count'] = len(new_objs)
        del new_objs

        if m2m_data:
            # get accession -> pk map
            qs = self.model._base_manager.values_list(
                *self.model.get_accession_lookups(),
                'pk',
            )
            accpk = {tuple(accs): pk for *accs, pk in qs}

            # replace accession with pk in m2m data keys
            m2m_data = {accpk[i]: data for i, data in m2m_data.items()}
            del accpk, qs

            # collecting all m2m entries
            for field in (i for i in fields if i.many_to_many):
                self._update_m2m(field.name, m2m_data)

        sleep(1)  # let progress meter finish before returning

        return diff, skipped_log

    def _update_m2m(self, field_name, m2m_data):
        """
        Update M2M data for one field -- helper for _load()

        :param str field_name: Name of m2m field
        :param dict m2m_data:
            A dict with all fields' m2m data as produced in the _load()
            method.
        """
        print(f'm2m {field_name}: ', end='', flush=True)
        field = self.model._meta.get_field(field_name)
        model = field.related_model

        # extract and flatten all accessions for field in m2m data
        accs = (
            i[0] for objdat in m2m_data.values()
            for i in objdat.get(field_name, [])
        )
        try:
            acc_field = model.get_accession_field_single()
        except AttributeError:
            acc_field = self.spec.extra[field_name]['acc_field']
        if acc_field.get_internal_type().endswith('IntegerField'):
            # cast to right type as needed (integers only?)
            accs = (int(i) for i in accs)
        accs = set(accs)
        print(f'{len(accs)} unique accessions in data', end='', flush=True)
        if not accs:
            print()
            return

        # get existing
        qs = model._base_manager.all()
        try:
            acc_lookup = model.get_accession_lookup_single()
        except AttributeError:
            acc_lookup = self.spec.extra[field_name]['acc_lookup']
        qs = qs.values_list(acc_lookup, 'pk')
        a2pk = dict(qs.iterator())
        print(f' / known: {len(a2pk)}', end='', flush=True)

        new_accs = [i for i in accs if i not in a2pk]
        if new_accs:
            # save additional remote side objects
            # NOTE: this will only work for those models for which the supplied
            # information (accession, source model and field) is sufficient,
            # might require overriding create_from_m2m_input().
            print(f' / new: {len(new_accs)}', end='')
            if a2pk:
                # if old ones exist, then display some of the new ones, this
                # is for debugging, in case these ought to be known
                print(' ', ', '.join([str(i) for i in islice(new_accs, 5)]),
                      '...')
            else:
                print()

            if hasattr(model, 'loader'):
                # assume here that we can't create these with only an accession
                print(f'WARNING: missing {model._meta.model_name} records can '
                      f'not be created here, use {model.__name__}.loader.load('
                      f')')
            else:
                model.objects.create_from_m2m_input(
                    new_accs,
                    source_model=self.model,
                    src_field_name=field_name,
                )

                # get updated m2m field's key -> pk mapping
                a2pk = dict(qs.iterator())
        else:
            print()

        # set relationships
        rels = []  # pairs of ours and other's PKs (+through params)
        for i, other in m2m_data.items():
            for j, *params in other[field_name]:
                try:
                    pk = a2pk[acc_field.to_python(j)]
                except KeyError:
                    # ignore for now
                    # FIXME
                    pass
                else:
                    rels.append((i, pk, *params))

        Through = field.remote_field.through  # the intermediate model
        if field.related_model == self.model:
            # m2m on self
            our_id_name = 'from_' + self.model._meta.model_name + '_id'
            other_id_name = 'to_' + model._meta.model_name + '_id'
        else:
            # m2m between two distinct models
            our_id_name = self.model._meta.model_name + '_id'
            other_id_name = model._meta.model_name + '_id'
        # extra fields for complex through models: take all, except the first
        # 3, which are the auto id and the FKs to and from, this must match the
        # given extra parameters
        extra_through_fields = [i.name for i in Through._meta.get_fields()[3:]]
        through_objs = [
            Through(
                **{our_id_name: i, other_id_name: j},
                **dict(zip(extra_through_fields, params))
            )
            for i, j, *params in rels
        ]

        # we don't know which objects are new and which got updated, so we're
        # deleting it all here so we can bulk create below
        print('Deleting existing m2m links for update... ', end='', flush=True)
        q = make_int_in_filter(our_id_name + '__pk', m2m_data.keys())
        qs = Through.objects.filter(q)
        delcount, _ = qs.delete()
        print(f'[{delcount} OK]')
        del q, qs

        self.bulk_create_wrapper(Through.objects.bulk_create)(through_objs)

    def _validate(self, obj, blank_invalid=True):
        """
        Validate an object.  This helper is called from _load() once per row.

        blank_invalid:
            If this is True, then we'll try to fix invalid data by setting the
            offending value to None

        Returns None if the given object is clean.  In some error cases it
        returns a dictionary listing the fields that got blanked/nulled if the
        resulting object was then clean.  Raises ValidationError in all other
        cases.
        """
        try:
            obj.full_clean()
        except ValidationError as e:
            if blank_invalid:
                blanked_log = []
                for key, errlist in e.error_dict.items():
                    try:
                        field = self.model._meta.get_field(key)
                    except FieldDoesNotExist:
                        continue
                    else:
                        if not field.null or not field.blank:
                            continue
                    for err in errlist:
                        if not isinstance(err, ValidationError):
                            continue
                        if err.code != 'invalid':
                            continue
                        # a bad value, e.g. text instead of a decimal
                        setattr(obj, field.name, None)
                        blanked_log.append(dict(
                            field=field.name,
                            value=err.params['value'],
                            message='invalid value, blanked/nulled',
                        ))
                if blanked_log:
                    try:
                        obj.full_clean()
                    except ValidationError:
                        pass
                    else:
                        return blanked_log

            # raise original error
            raise e

    def get_choice_value_prep_method(self, field):
        """
        Return pre-processing method for choice value prepping
        """
        prep_values = {j: i for i, j in field.choices}

        def prep_choice_value(self, value, row=None, obj=None):
            """ get the prepared field value for a choice field """
            return prep_values.get(value, value)

        return partial(prep_choice_value, self)

    def split_m2m_value(self, value, row=None, sep=None):
        """
        Pre-processor to split list-field values

        sep:
            If this is None, then commas or semi-colons are recognized as
            separators.  A ValueError is raised if a comma and semi-colon bot
            appear in the value.

        This will additionally strip leading or trailing white-space, ignore
        extra separators (meaning blanks can not appear as a listed value), and
        sort and remove duplicates.  If you don't want this use the
        split_m2m_value_simple() method.
        """
        if value is None:
            return []

        if sep is None:
            if ',' in value and ';' in value:
                raise ValueError('ambigious separators')
            value = value.replace(',', ';')
            sep = ';'
        items = (val for i in value.split(sep) if (val := i.strip()))
        return sorted(set(items))

    def split_m2m_value_simple(self, value, row=None):
        """
        Pre-processor to split semi-colon-separated list-field values
        """
        return value.split(';')

    def iterate_rows(self, rows, start=0):
        """
        Helper to advance over rows manage the current row's data

        This generator will yield the current row's number
        """
        get_row_data = self.spec.row_data  # get a local ref
        for i, row in enumerate(rows, start=start):
            self.current_lineno = i
            self.current_row = row
            self.current_row_data = list(get_row_data(row))
            yield i

    def get_current_value(self, field_name):
        """ Return a value from the current row """
        try:
            for field, _, value in self.current_row_data:
                if field.name == field_name:
                    return value
        except AttributeError:
            if not hasattr(self, 'current_row_data'):
                # iterate_rows() needs to set up current_row_data first
                raise RuntimeError('iterate_rows() must be called first')
            raise  # something else going on

        # for debugging:
        print(f'\n[DEBUG] dumping {self}.current_row_data: ')
        for n, i in enumerate(self.current_row_data, start=1):
            print('   ', n, i)
        raise LookupError(f'no such field in row data: {field_name}')

    @atomic_dry
    def fast_bulk_update(self, objs, fields, batch_size=None):
        """
        A faster alternative for bulk_update

        This does not use QuerySet.update() machinery that bulk_update benefits
        from, so there are some limitations:

        * no support for dealing multi-table inheritance

        On the other hand this can run much faster and can handle many update
        fields just fine.
        """
        if connection.vendor == 'postgresql':
            meth = self._fast_bulk_update_batch
            if batch_size is None:
                batch_size = 20000

        elif connection.vendor == 'sqlite':
            meth = self._fast_bulk_update_batch
            if batch_size is None:
                # limit batch size to maximum allowed variable
                batch_size = \
                    settings.SQLITE_MAX_VARIABLE_NUMBER // (len(fields) + 1)
        else:
            return self.bulk_update(objs, fields, batch_size=batch_size)

        return self.bulk_update_wrapper(meth)(objs, fields, batch_size)

    def _parse_rows(self, rows, parse_into):
        for _ in self.iterate_rows(rows):
            rec = {}
            for field, _, value in self.current_row_data:
                if field.many_to_many and value is not None:
                    value = self.split_m2m_value(value)

                rec[field.name] = value

            parse_into.append(rec)

    def _get_objects_for_update(self, template):
        """
        helper for load() to get a pool of objects that can be updated

        Returns a dictionary mapping accession keys to instances.  The keys are
        the accession fields less the fields in the template.  Because of the
        way attrgetter works, if there is a single accession field used, then
        the key will be a single scalar value and if muliple fields are used
        then then the key will be a tuple of values.  This needs to be taken
        into account if conversion functions are used to mangle the field's
        value to match the keys.
        """
        # 1. get the key(s) with which to index the pool
        key_fields = [
            i for i in self.model.get_accession_fields()
            if i.name not in template
        ]
        # use attname, so for FKs we index by PK
        get_pool_key = attrgetter(*[i.attname for i in key_fields])
        # so the key is either a scalar value or a tuple of values

        qs = self.filter(**template)

        obj_pool = {}
        for i in qs:
            key = get_pool_key(i)
            if key in obj_pool:
                # Field should be unique (after any template filter), but maybe
                # nulls are allowed, anyways, can't let this go through here
                raise RuntimeError(f'duplicate key: {key=} {len(obj_pool)=}')
            obj_pool[key] = i
        return obj_pool

    def _fast_bulk_update_batch(self, objs, field_names):
        """
        Fast-update a single batch

        Parameters:
            objs: an iterable of objects to be updated
            field_names: a list of str, names of the fields to be updated

        This implements a four step process that should work with sqlite3 and
        postgresql:

            1. create a temporary table
            2. insert data and primary keys into the temp table
            3. update real table from temp table
            4. drop the temp table again

        The column definitions for the temporary table may be missing some
        constraint bits, e.g. decimal places for the decimal fields.  Is it not
        entirely clear whether this may cause data loss.

        The generated SQL has one variable per updated field plus primary key
        per object.  Sqlite3 has a compiled-in SQLITE_MAX_VARIABLE_NUMBER value
        that limits how many variables may be include. This limits the number
        of objects that can be processed.  The relevant batching must be
        managed by the caller.
        """
        TEMP_TABLE = 'temp_fast_bulk_update_table'
        pk_field = self.model._meta.pk
        fields = [pk_field] + [
            self.model._meta.get_field(i)
            for i in field_names
        ]
        for i in fields:
            if i not in self.model._meta.local_concrete_fields:
                # wrong field name or m2m or model inheritance
                raise RuntimeError(f'not local+concrete: {i} of {self.model}')
        col_defs = ', '.join([
            f'{i.column} {i.db_type(connection)}'
            for i in fields
        ])
        create_sql = f'CREATE TEMP TABLE {TEMP_TABLE} ({col_defs})'

        col_names = ', '.join([i.column for i in fields])
        # placeholders & parameters
        # (there can be lots and lots of them, up to SQLITE_MAX_VARIABLE_NUM)
        insert_params = []
        field_n_attrs = tuple(((i, i.attname) for i in fields))
        for obj_count, i in enumerate(objs, start=1):
            for field, attr in field_n_attrs:
                insert_params.append(
                    field.get_db_prep_save(getattr(i, attr), connection)
                )
        num_fields = len(fields)
        values = ','.join(['({})'.format(','.join(['%s'] * num_fields))] * obj_count)  # noqa: E501

        insert_sql = f'INSERT INTO {TEMP_TABLE} ({col_names}) VALUES {values}'
        table = self.model._meta.db_table
        set_expr = ', '.join([
            f'{i.column} = {TEMP_TABLE}.{i.column}'
            for i in fields
        ])
        cond = f'{table}.{pk_field.column} = {TEMP_TABLE}.{pk_field.column}'
        update_sql = \
            f'UPDATE {table} SET {set_expr} FROM {TEMP_TABLE} WHERE {cond}'
        with transaction.atomic(), connection.cursor() as cur:
            cur.execute(create_sql, [])
            cur.execute(insert_sql, insert_params)
            cur.execute(update_sql, [])
            cur.execute(f'DROP TABLE {TEMP_TABLE}', [])

    def quick_erase(self):
        quickdel = import_string(
            'mibios.umrad.model_utils.delete_all_objects_quickly'
        )
        quickdel(self.model)


class Loader(BulkCreateWrapperMixin, BaseLoader.from_queryset(QuerySet)):
    pass


class MetaDataLoader(Loader):
    """ Loader class for small amounts of data with more checks """
    default_load_kwargs = dict(
        validate=True,
        bulk=False,
        update=True,
        diff_stats=True,
    )


class BulkLoader(Loader):
    def load(self, bulk=True, validate=False, **kwargs):
        super().load(bulk=bulk, validate=validate, **kwargs)


class CompoundRecordLoader(BulkLoader):
    """ loader for compound data from MERGED_CPD_DB.txt file """

    def get_file(self):
        return settings.UMRAD_ROOT / 'MERGED_CPD_DB.txt'

    def chargeconv(self, value, obj):
        """ Pre-processor to convert '2+' -> 2 / '2-' -> -2 """
        if value == '' or value is None:
            return None
        elif value.endswith('-'):
            return -int(value[:-1])
        elif value.endswith('+'):
            return int(value[:-1])
        else:
            try:
                return int(value)
            except ValueError as e:
                raise InputFileError from e

    def collect_others(self, value, obj):
        """
        Pre-processor triggered on kegg column to collect from other columns
        """
        # assume that value == current_row[7]
        lst = self.split_m2m_value(';'.join(self.current_row[7:12]))
        try:
            # remove the record itself
            lst.remove(self.current_row[0])
        except ValueError:
            pass
        return ';'.join(lst)

    spec = CSV_Spec(
        ('cpd', 'accession'),
        ('formula', 'formula'),
        ('mass', 'mass'),
        ('charge', 'charge', chargeconv),
        ('db_src', 'source'),
        ('tcdbs', None),
        ('names', 'names'),
        ('keggcpd', 'others', collect_others),
        ('chebcpd', None),  # remaining columns processed by kegg
        ('hmdbcpd', None),
        ('pubccpd', None),
        ('inchcpd', None),
        ('bioccpd', None),
    )


class FuncRefDBEntryLoader(BulkLoader):
    """
    Loads data from Function_Names.txt

    Run load() after UniRef100 data is loaded.  May add to function name table
    and add links between function cross refs and names.
    """
    spec = CSV_Spec('accession', 'names')

    def get_file(self):
        return settings.UMRAD_ROOT / 'Function_Names.txt'

    @atomic_dry
    def load(self, dry_run=False):
        field = self.model._meta.get_field('names')  # the m2m field
        FunctionName = field.related_model

        xref2pk = {i.accession: i.pk for i in self.model.objects.iterator()}
        known_names = set(FunctionName.objects.values_list('entry', flat=True))

        new_names = set()
        data = {}
        unknown_xrefs = set()

        with self.get_file().open() as f:
            f = ProgressPrinter(f'lines read from {f.name}')(f)
            for line in f:
                xref, names = line.rstrip('\n').split('\t')
                names = names.split(';')
                if xref in data:
                    raise InputFileError(f'duplicate: {line}')

                pk = xref2pk.get(xref, None)
                if pk is None:
                    unknown_xrefs.add(xref)
                    continue
                for i in names:
                    if i not in known_names:
                        new_names.add(i)
                data[pk] = {'names': (names, )}  # mimic m2m_data in super load

        if unknown_xrefs:
            print(f'WARNING: found {len(unknown_xrefs)} unknown function xrefs'
                  f': {[i for _, i in zip(range(5), unknown_xrefs)]}')

        print(f'func names known: {len(known_names)}, new: {len(new_names)}')
        if new_names:
            name_objs = (FunctionName(entry=i) for i in new_names)
            FunctionName.objects.bulk_create(name_objs)

        self._update_m2m('names', data)


class ReactionRecordLoader(BulkLoader):
    """ loader for reaction data from MERGED_RXN_DB.txt file """

    def contribute_to_class(self, model, name):
        super().contribute_to_class(model, name)
        # init value prep for ReactionCompound field choices
        ReactionCompound = \
            self.model._meta.get_field('compound').remote_field.through
        field = ReactionCompound._meta.get_field('side')
        self._prep_side_val = self.get_choice_value_prep_method(field)
        field = ReactionCompound._meta.get_field('location')
        self._prep_loc_val = self.get_choice_value_prep_method(field)
        field = ReactionCompound._meta.get_field('transport')
        self._prep_trn_val = self.get_choice_value_prep_method(field)

        self.loc_values = [i[0] for i in ReactionCompound.LOCATION_CHOICES]

    def get_file(self):
        return settings.UMRAD_ROOT / 'MERGED_RXN_DB.txt'

    def errata_check(self, value, obj):
        """ Pre-processor to skip extra header rows """
        # FIXME: remove this function when source data is fixed
        if value == 'rxn':
            raise SkipRow('value == "rxn" (see source code)')
        return value

    def process_xrefs(self, value, obj):
        """
        Pre-processor to collect data from all xref columns

        subsequenc rxn xref columns will then be skipped
        Returns list of tuples.
        """
        row = self.current_row
        xrefs = []
        for i in row[3:6]:  # the alt_* columns
            xrefs += self.split_m2m_value(i)
        xrefs = [(i, ) for i in xrefs if i != row[0]]  # rm this rxn itself
        return xrefs

    def process_compounds(self, value, obj):
        """
        Pre-processor to collect data from the 18 compound columns

        Will remove duplicates.  Other compound columns should be skipped
        """
        row = self.current_row
        all_cpds = []
        # outer loop: source values in order of column sets in input file
        for i, src in enumerate(('CH', 'KG', 'BC')):
            for j, side in enumerate(self.loc_values):
                # excl. first 6, then take sets of 3
                # i in {0,1,2} and j in {0,1}
                offs = 6 + 6 * i + 3 * j
                cpds, locs, trns = row[offs:offs + 3]

                cpds = self.split_m2m_value_simple(cpds)
                locs = self.split_m2m_value_simple(locs)
                trns = self.split_m2m_value_simple(trns)
                if not (len(cpds) == len(locs) == len(trns)):
                    raise InputFileError(
                        'inconsistent compound data list lengths: '
                        f'{row[offs:offs + 3]=} '
                        f'{cpds=} {locs=} {trns=}'
                    )

                uniq_cpds = set()
                for c, l, t in zip(cpds, locs, trns):
                    cpd_dat = (
                        c,
                        src,
                        self._prep_side_val(side),
                        self._prep_loc_val(l),
                        self._prep_trn_val(t),
                    )
                    if c in uniq_cpds:
                        if cpd_dat in all_cpds:
                            # skip duplicate
                            continue
                        else:
                            # FIXME -- are bad dups a bug or a feature?
                            # raise InputFileError(
                            print(
                                f'ERROR: (SKIPPING BAD DUPLICATE (FIXME)) '
                                f'@row: {row[0]} '
                                f'compound {c} occurs multiple times but '
                                f'loc/trn data does not agree'
                            )
                            continue
                    else:
                        uniq_cpds.add(c)

                    all_cpds.append(cpd_dat)

        return all_cpds

    spec = CSV_Spec(
        ('rxn', 'accession', errata_check),
        ('db_src', 'source'),
        ('rxn_dir', 'direction'),
        ('alt_rhea', 'others', process_xrefs),
        ('alt_kegg', None),
        ('alt_bioc', None),
        ('rhea_lcpd', 'compound', process_compounds),
        ('rhea_lloc', None),
        ('rhea_ltrn', None),
        ('rhea_rcpd', None),
        ('rhea_rloc', None),
        ('rhea_rtrn', None),
        ('kegg_lcpd', None),
        ('kegg_lloc', None),
        ('kegg_ltrn', None),
        ('kegg_rcpd', None),
        ('kegg_rloc', None),
        ('kegg_rtrn', None),
        ('bioc_lcpd', None),
        ('bioc_lloc', None),
        ('bioc_ltrn', None),
        ('bioc_rcpd', None),
        ('bioc_rloc', None),
        ('bioc_rtrn', None),
        ('UPIDs', 'uniprot'),
        ('ECs', 'ec'),
    )


class UniRef100Loader(BulkLoader):
    """ loader for OUT_UNIREF.txt """

    empty_values = ['N/A']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        FuncRefDBEntry = import_string('mibios.umrad.models.FuncRefDBEntry')
        self.func_dbs = FuncRefDBEntry.DB_CHOICES

    def get_file(self):
        """ get path to UNIREF100_INFO file made by Create_Alignment_DB.pl """
        return settings.UMRAD_ROOT / 'UNIREF100_INFO.txt'

    @atomic_dry
    def load(self, selection_from=None, **kwargs):
        """
        Populate UniRef100 from INFO file

        :params:
            selection_from: path to text file listing UniRef100 accessions.
            Such a file can be compiled via
            mibios.omics.Sample.loader.ur100_accession_crawler().
        """
        self.selected_accns = None
        if selection_from:
            with open(selection_from) as ifile:
                self.selected_accns = {
                    self.parse_ur100(line.strip())
                    for line in ifile
                }
            print(f'Selectively loading {len(self.selected_accns)} UR100 '
                  f'accessions.')

        # get map old->new for use by merge_taxids pre-processor
        print('Loading merged taxnodes... ', end='', flush=True)
        MergedNodes = import_string('mibios.ncbi_taxonomy.models.MergedNodes')
        self.merged = dict(
            MergedNodes.objects
            .select_related('new_node__taxid')
            .values_list('old_taxid', 'new_node__taxid')
        )
        print(f'[{len(self.merged)} OK]')
        self.funcref2db = {}
        super().load(**kwargs)
        # set DB values for func xrefs -- can't do this in the regular load
        # as there we only have the accession to work with with m2m-related
        # objects
        FuncRefDBEntry = import_string('mibios.umrad.models.FuncRefDBEntry')
        a2objs = FuncRefDBEntry.objects.in_bulk(field_name='accession')
        objs = []
        for acc, db in self.funcref2db.items():
            obj = a2objs[acc]
            obj.db = db
            objs.append(obj)
        pp = ProgressPrinter('func xrefs db values assigned')
        FuncRefDBEntry.name_loader.fast_bulk_update(pp(objs), ['db'])

    @classmethod
    def parse_ur100(cls, value):
        """ helper to parse UniRef100 accessions """
        return value.upper().removeprefix('UNIREF100_')

    def parse_and_filter(self, value, obj):
        """ for selective loading, check if row is to be skipped """
        value = self.parse_ur100(value)
        if self.selected_accns is not None:
            if value not in self.selected_accns:
                raise SkipRow('record was not selected by caller', log=False)
        return value

    def process_func_xrefs(self, value, obj):
        """ Pre-processor ro collect COG through EC columns """
        ret = []
        for (db_code, _), vals in zip(self.func_dbs, self.current_row[13:19]):
            for i in self.split_m2m_value(vals):
                try:
                    db = self.funcref2db[i]
                except KeyError:
                    self.funcref2db[i] = db_code
                else:
                    # consistency check
                    if db != db_code:
                        raise RuntimeError('func xref db inconsistency')
                ret.append((i, ))
        return ret

    def process_reactions(self, value, obj):
        """ Pre-processor to collect all reactions """
        rxns = set()
        for i in self.current_row[17:20]:
            items = self.split_m2m_value(i)
            for j in items:
                if j in rxns:
                    raise InputFileError('reaction accession dupe')
                else:
                    rxns.add(j)
        return [(i, ) for i in rxns]

    def merge_taxids(self, value, obj):
        """ replace any merged taxid with new taxid """
        items = self.split_m2m_value(value)
        if not items:
            return value
        try:
            taxids = [int(i) for i in items]
        except ValueError as e:
            raise InputFileError(f'failed parsing taxid: {e}') from e

        # merge and remove any introduced (and otherwise) duplicates
        taxids = set((self.merged.get(i, i) for i in taxids))
        # this is an m2m field, so return list-of-tuples
        return [(i, ) for i in taxids]

    spec = CSV_Spec(
        ('UR100', 'accession', parse_and_filter),
        ('UR90', 'uniref90'),
        ('Name', 'function_names'),
        ('Length', 'length'),
        ('SigPep', 'signal_peptide'),
        ('TMS', 'tms'),
        ('DNA', 'dna_binding'),
        ('TaxonId', 'taxa', merge_taxids),
        ('Binding', 'binding'),
        ('Loc', 'subcellular_locations'),
        ('TCDB', 'function_refs', process_func_xrefs),
        ('COG', None),
        ('Pfam', None),
        ('Tigr', None),
        ('Gene_Ont', None),
        ('InterPro', None),
        ('ECs', None),
        ('kegg', 'reactions', process_reactions),
        ('rhea', None),
        ('biocyc', None),
    )


class BaseManager(MibiosBaseManager):
    """ Manager class for UMRAD data models """
    def create_from_m2m_input(self, values, source_model, src_field_name):
        """
        Store objects from accession or similar value (and context, as needed)

        :param list values:
            List of accessions
        :param Model source_model:
            The calling model, the model on the far side of the m2m relation.
        :param str src_field_name:
            Name of the m2m field pointing to us

        Inheriting classes should overwrite this method if more that the
        accession or other unique ID field is needed to create the instances.
        The basic implementation will assign the given value to the accession
        field, if one exists, or to the first declared unique field, other than
        the standard id AutoField.

        This method is responsible to set all required fields of the model,
        hence the default version should only be used with controlled
        vocabulary or similarly simple models.
        """
        # TODO: if need arises we can implement support for values to
        # contain tuples of values
        attr_name = self.model.get_accession_lookup_single()
        objs = (self.model(**{attr_name: i}) for i in values)
        return self.bulk_create(objs)


class Manager(BulkCreateWrapperMixin, BaseManager.from_queryset(QuerySet)):
    pass
