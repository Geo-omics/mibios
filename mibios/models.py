from collections import namedtuple, OrderedDict
from functools import cache
from itertools import groupby
import json
from operator import itemgetter
from pathlib import Path
from shutil import copy2

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core import serializers
from django.core.files import File
from django.core.management import call_command
from django.urls import reverse
from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import models, transaction
from django.db.migrations.recorder import MigrationRecorder
from django.db.utils import DEFAULT_DB_ALIAS, ConnectionHandler
from django.utils.html import format_html
from rest_framework.serializers import HyperlinkedModelSerializer
from rest_framework.viewsets import ReadOnlyModelViewSet

from . import get_registry
from .fields import AutoField
from .managers import CurationManager, Manager
from .query import QuerySet
from .utils import getLogger


log = getLogger(__name__)


class NaturalKeyLookupError(Exception):

    """
    Raised when a natural lookup fails to resolve

    Handle like a user input error, but beware they might be bugs in the
    natural lookup handling code
    """
    pass


Fields = namedtuple('Fields', ['fields', 'names', 'verbose'])
""" container to hold list of fields for a model """


class ImportFile(models.Model):
    """
    Represents the imported files
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    name = models.CharField(
        max_length=300, verbose_name='original filename',
        help_text='This is the original, human-readable name for this file. '
                  'It is allowed to have the same name for multiple imported '
                  'file records.  This practice is encouraged e.g. if the '
                  'same file is uploaded multiple times, e.g. re-uploading '
                  'after making offline changes to the file.',
    )
    file = models.FileField(
        upload_to='imported/%Y/', verbose_name='data source file',
    )
    log = models.TextField(
        blank=True, default='', help_text='log output from data import',
    )
    note = models.TextField(blank=True, default='', help_text='File meta '
                            'info: origin of data, why was it imported etci.')

    class Meta:
        get_latest_by = 'timestamp'
        ordering = ['-timestamp']
        verbose_name = 'uploaded and imported file'

    def __str__(self):
        return '{} {}'.format(self.timestamp, self.name)

    @classmethod
    def create_from_file(cls, file, **kwargs):
        """
        Create a new instance from given file-like object

        This will take the last component of the path as the original filename
        and remove path components before storiung the file.

        :param file: Either a str or pathlib.Path containing the (original)
                     path to the file or a file-like object.
        :param dict kwargs: Additional instance attributes to support
                            subclasses
        """
        need_close = True
        if isinstance(file, str):
            file = open(file)
        elif isinstance(file, Path):
            file = file.open()
        else:
            # assume file-like obj, caller must call close
            need_close = False

        name = Path(file.name).name
        file = File(file, name=name)
        obj = cls.objects.create(name=name, file=file, **kwargs)
        if need_close:
            file.close()
        return obj

    def get_log_url(self):
        """
        Return URL to log view as HTML-safe string
        """
        if not self.log:
            return None
        templ = '<a href="{}">view</a>'
        url = reverse('log', kwargs=dict(import_file_pk=self.pk))
        return format_html(templ, url)
    get_log_url.short_description = 'Import log'

    def get_abbr_note(self, import_file=None, max_length=75):
        """
        Return an abbreviated form of the note

        Returns the first sentence of the first line of the note, at most
        max_length characters.
        """
        if not self.note:
            return ''

        txt = self.note[:max_length].splitlines()[0]
        txt = txt.split('.', maxsplit=1)[0]
        return txt


class ChangeRecordQuerySet(QuerySet):
    def with_old_fields(self):
        """
        Add fields from previous change record

        This annotates each instance with an "old_fields" text field that is
        populated from the record's preceding change's "fields" field.
        ChangeRecord.diff() can use this to make the diff.
        """
        clone = self._chain()
        # older: sub query to get previous changes for object
        older = ChangeRecord.objects.filter(
            record_type=models.OuterRef('record_type'),
            record_pk=models.OuterRef('record_pk'),
            timestamp__lte=models.OuterRef('timestamp'),
        ).order_by('-timestamp')
        # older includes the changes itself, sp ick the second change in order
        old_fields = models.Subquery(
            older.values('fields')[1:2],
            output_field=models.TextField(blank=True)
        )
        clone = clone.annotate(old_fields=old_fields)
        return clone


class ChangeRecord(models.Model):
    """
    Model representing a changelog entry
    """
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )
    file = models.ForeignKey(ImportFile, on_delete=models.PROTECT, null=True)
    line = models.IntegerField(
        null=True, blank=True,
        help_text='The corresponding line in the input file',
    )
    comment = models.CharField(
        max_length=200, blank=True,
        help_text='Additional info, comment, or management command for import',
    )
    record_type = models.ForeignKey(
        ContentType, on_delete=models.SET_NULL, null=True, blank=True,
    )
    record_pk = models.PositiveIntegerField(null=True, blank=True)
    record = GenericForeignKey('record_type', 'record_pk')
    record_natural = models.CharField(max_length=300, blank=True,
                                      db_index=True)
    fields = models.TextField(blank=True)
    is_created = models.BooleanField(default=False, verbose_name='new record')
    is_deleted = models.BooleanField(default=False)

    class Meta:
        get_latest_by = 'timestamp'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=('record_type', 'record_pk')),
        ]

    objects = Manager.from_queryset(ChangeRecordQuerySet)()

    def __str__(self):
        user = ' ' + self.user.username if self.user else ''

        if self.is_deleted:
            return '{}{} (deleted) - {} (pk:{})'.format(
                self.timestamp, user, self.record_natural, self.record_pk
            )

        new = ' (new)' if self.is_created else ''
        dots = '...' if len(self.fields) > 80 else ''
        return '{}{}{} - {}{}'.format(self.timestamp, user, new,
                                      self.fields[:80], dots)

    def has_changed(self):
        """
        Compare with previous change record of same object
        """
        qs = self.record.history
        if self.timestamp is not None:
            qs = qs.filter(timestamp__lt=self.timestamp)

        try:
            prev = qs.latest()
        except self.DoesNotExist:
            # we are first
            return True

        if self.record_natural != prev.record_natural:
            return True

        if not self.fields:
            self.serialize()

        if self.fields != prev.fields:
            return True

        return False

    def serialize(self):
        """
        Serialize field content
        """
        self.fields = serializers.serialize(
            'json',
            [self.record],
            fields=self.record.get_fields(skip_auto=True, with_m2m=True).names,
            use_natural_foreign_keys=True
        )

    def save(self, *args, **kwargs):
        """
        Save change record

        Before saving, serialize the record.  After saving we remove the change
        record from the data record to ensure that for subsequent changes a new
        change records is created.
        """
        self.serialize()
        super().save(*args, **kwargs)
        self.record.history.add(self)
        if hasattr(self.record, 'change'):
            del self.record.change

    def fields_as_dict(self, json_serial=None):
        """
        Return serialized object as dict

        :param str json_serial:
            Alternative json serialization string.  The default is to use the
            instance's "fields" attribute.

        Extract the fields as a dict, primary key is included as "pk".
        """
        if json_serial is None:
            json_serial = self.fields
        serial = json.loads(json_serial)[0]
        fields = serial['fields']
        if 'pk' not in fields:
            fields['pk'] = serial['pk']
        return fields

    def get_predecessor(self, constant_pk=True):
        """
        Return previous change of object.

        Returns None if this is the first change record for given object.

        :param bool constant_pk: If True, then the precessor is guarrantied to
                                 have the same primary key, but may have a
                                 different natural key.  The default behavior
                                 is to pick the precessor with the same primary
                                 key, allowing for changes of the natural key.
        """
        f = dict(record_type=self.record_type, timestamp__lte=self.timestamp)
        if constant_pk:
            f['record_pk'] = self.record_pk
        else:
            f['record_natural'] = self.record_natural

        qs = ChangeRecord.objects.filter(**f).order_by('-timestamp')
        try:
            return (qs[1:2]).get()
        except ChangeRecord.DoesNotExist:
            return None

    def diff_to(self, other=None):
        """
        Return diff comparing fields to given other change record
        """
        if other is None:
            other = self.get_predecessor()

        if other is None:
            # no predecessor
            fields = {}
        else:
            fields = other.fields_as_dict()
            if 'name' not in fields:
                fields['name'] = other.record_natural
        return self.diff(fields)

    def diff(self, theirs=None):
        """
        Get the difference in field values introduced by this change

        :param dict theirs: Compare against these fields.  The default is
                                to use the fields in the old_fields instance
                                attribute.

        Fields that were dropped from the tables between the changes will not
        be listed.
        """
        if theirs is None:
            if hasattr(self, 'old_fields'):
                if self.old_fields is None:
                    theirs = {}
                else:
                    theirs = self.fields_as_dict(self.old_fields)
            else:
                return self.diff_to(other=None)

        ours = self.fields_as_dict()

        if 'name' in theirs and 'name' not in ours:
            ours['name'] = self.record_natural

        diff = {}
        for k, v in ours.items():
            try:
                old_v = theirs[k]
            except KeyError:
                diff[k] = (v,)
            else:
                if v != old_v:
                    diff[k] = (old_v, v)
        return diff

    @classmethod
    def summary(cls):
        """
        Generator of compact history table

        Starts with mose recent change.
        """
        users = User.objects.in_bulk()
        rec_types = ContentType.objects.in_bulk()
        files = ImportFile.objects.in_bulk()

        group_key = ('user_id', 'file_id', 'record_type_id', 'comment')
        qs = ChangeRecord.objects.values('id', 'timestamp', *group_key)
        g = groupby(qs.iterator(), key=itemgetter(*group_key))

        for (user_id, file_id, record_type_id, comment), grp in g:
            grp = list(grp)
            record_type = rec_types.get(record_type_id)
            file = files.get(file_id)
            user = users.get(user_id)
            yield (
                # order of items as expected by .tables.CompactHistoryTable
                # and summary_shorter()
                (grp[-1]['id'], grp[0]['id']),  # smaller pk goes first
                grp[0]['timestamp'],
                len(grp),
                record_type,
                comment or (file.get_abbr_note() if file else ''),
                file,
                user or 'admin',
            )

    @classmethod
    def summary_shorter(cls, limit=None):
        """
        A more concise change summary

        Collapses summary rows that differ by less than <...> and comments

        :param int limit: Limit to at most this many compacted rows
        """
        comment = '(see details)'
        keys = ['ids', 'ts', 'count', 'rec_t', 'comment', 'file', 'user']
        buf = None
        nrows = 0

        # iterate from last to first change set:
        for row in cls.summary():
            n = dict(zip(keys, row))

            if buf is None:
                buf = dict(zip(keys, row))
                continue

            combine = (
                0 <= (buf['ts'] - n['ts']).seconds < 30
                and n['rec_t'] == buf['rec_t']
                and n['file'] == buf['file']
                and n['user'] == buf['user']
            )
            if combine:
                buf['ids'] = (n['ids'][0], buf['ids'][1])  # update int start
                buf['ts'] = n['ts']  # use earlier ts
                buf['count'] += n['count']
                buf['comment'] = comment
            else:
                yield tuple(buf.values())
                nrows += 1
                if limit and nrows >= limit:
                    buf = None  # don't yield buf after break
                    break
                buf = n

        if buf is not None:
            yield tuple(buf.values())

    @classmethod
    def summary_dict(cls, **kwargs):
        """
        Compact history tables with rows as dicts

        This is to support CompactHistoryView with django_tables2.Table
        consumption.
        """
        keys = ['details', 'timestamp', 'count', 'record_type', 'comment',
                'file', 'user']

        return [dict(zip(keys, i)) for i in cls.summary_shorter(**kwargs)]

    @classmethod
    def get_details(cls, first, last):
        """
        Get queryset for DetailedHistoryView

        :param int first: lowest primary key of range of changes to show
        :param int last: highest primary key of range of changes to show

        Returns all change records in given range with old field annotation.
        """
        return (cls.objects
                .filter(pk__gte=first, pk__lte=last)
                .select_related('user', 'record_type', 'file')
                .with_old_fields())

    def format(self):
        """
        Pretty-print object

        Returns a formatted string.
        """
        if self.is_created:
            mode = 'new'
        elif self.is_deleted:
            mode = 'deleted'
        else:
            mode = 'modified'
        out = (
            f'Change {self.pk}: {self.timestamp} ({self.user})\n'
            f'file: {self.file} line: {self.line}\n'
            f'comment: {self.comment}\n'
            f'{mode} record: {self.record} ({self.record_natural}) type: '
            f'{self.record_type} pk: {self.record_pk}\n'
            f'fields:\n'
        )
        fields = self.fields_as_dict()
        col_width = max((len(i) for i in fields.keys())) + 1

        if mode == 'modified':
            diff = self.diff()
            fields = {
                k: f'{diff[k][0]} => {v}' if k in diff else v
                for k, v in fields.items()
            }

        for k, v in fields.items():
            out += f'{k.rjust(col_width)}: {v}\n'

        return out


def _default_snapshot_name():
    try:
        last_pk = Snapshot.objects.latest().pk
    except Snapshot.DoesNotExist:
        last_pk = 0

    if hasattr(get_registry(), 'name'):
        name = get_registry().name
    else:
        name = Path(settings.DATABASES['default']['NAME']).stem
    return name + ' version ' + str(last_pk + 1)


class Snapshot(models.Model):
    """
    Snapshot of database

    An instance will make a copy the first time it is saved.  After that, the
    name and the note can be edited.  The snapshot consists of a copy of the
    sqlite3 database file and a json dump of the models in the apps in the
    mibios registry.  Deleting the object from the database will also delete
    the snapshot files.
    """
    def get_app_list():
        """
        Helper to generate the content of the app_list field
        """
        return ','.join(get_registry().apps.keys())

    timestamp = models.DateTimeField(auto_now_add=True)
    name = models.CharField(
        max_length=100, default=_default_snapshot_name, unique=True,
    )
    app_list = models.CharField(
        max_length=1000,
        editable=False,
        default=get_app_list,
        help_text='Comma-separated list of names of the apps that have data '
                  'tables stored in the snapshot.'
    )
    # migrations: keeps the last applied migration for each registered app, we
    # could put a foreign key to MigrationRecorder.Migration here but the check
    # framework will complain, as the migration model is not fully integrated
    # into everything. This may help in the future to maybe use the ORM to
    # access the snapshots.
    migrations = models.CharField(
        max_length=3000,
        default='',
        editable=False,
        help_text='json serializaton of the last migrations of each app',
    )
    dbfile = models.CharField(
        max_length=500,
        editable=False,
        verbose_name='archived database file',
    )
    jsondump = models.CharField(
        max_length=500,
        editable=False,
        verbose_name='JSON formatted archive',
    )
    note = models.TextField(blank=True)

    class Meta:
        get_latest_by = 'timestamp'
        ordering = ['-timestamp']
        verbose_name = 'database version'

    def __str__(self):
        return self.name

    def dbpath(self):
        return settings.SNAPSHOT_DIR / self.dbfile
    dbpath.short_description = 'path to sqlite3 db file'

    def jsonpath(self):
        return settings.SNAPSHOT_DIR / self.jsondump
    jsonpath.short_description = 'path to json dump file'

    def save(self, *args, **kwargs):
        if not self.pk:
            # only take snapshot when saving instance for first time
            self._create_snapshot()
        super().save(*args, **kwargs)

    def _create_snapshot(self):
        if not settings.SNAPSHOT_DIR.is_dir():
            settings.SNAPSHOT_DIR.mkdir(mode=0o770, parents=True)
        src = settings.DATABASES['default']['NAME']
        stem = '_'.join(self.name.split())

        self.dbfile = stem + '.sqlite3'
        self.jsondump = stem + '.json'

        copy2(src, str(self.dbpath()))

        call_command(
            'dumpdata',
            *get_registry().apps,
            format='json',
            indent=4,
            database='default',
            natural_foreign=True,
            natural_primary=True,
            output=self.jsonpath(),
        )

        # set read-only
        self.dbpath().chmod(0o440)
        self.jsonpath().chmod(0o440)

        # migration state
        migrations = []
        for i in ['mibios'] + list(get_registry().apps.keys()):
            migrations.append(
                MigrationRecorder.Migration.objects
                .filter(app=i)
                .order_by('applied')
                .last()
            )

        self.migrations = serializers.serialize('json', migrations)

    def delete(self, *args, **kwargs):
        self.dbpath().unlink(missing_ok=True)
        self.jsonpath().unlink(missing_ok=True)
        super().delete(*args, **kwargs)

    def do_sql(self, sql, params=[], descr=False):
        """
        Connect to snapshot db, run sql and fetchall rows
        """
        db = {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': 'file:{}?mode=ro'.format(self.dbpath()),
            'OPTIONS': {'uri': True, },
        }
        conf = {DEFAULT_DB_ALIAS: db}
        conn_h = ConnectionHandler(conf)
        conn = conn_h[DEFAULT_DB_ALIAS]
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception:
            raise
        else:
            if descr:
                return rows, cur.description
            else:
                return rows
        finally:
            conn.close()

    def get_table_names(self, app_label, app_check=True):
        """
        Get list of the snapshot's full table names for given app

        :param str app_labels: App label
        :param bool app_check: Ensure to only return names of data app tables
        """
        if app_check:
            if app_label not in self.app_list.split(','):
                raise ValueError(f'is not a data app in snapshot: {app_label}')

        sql = ('select name from sqlite_master where '
               'type = "table" and name like %s '
               'and name not like "%%_history"')

        pat = app_label + '_%'
        rows = self.do_sql(sql, [pat])
        return [i[0] for i in rows]

    def get_table_name_data(self, *app_labels):
        """
        Get list of the snapshot's table names as dict

        If not app labels are providen, then all data tables are returned.  Two
        columns are returned, the app label and the short table name.  The
        short table name has the app label removed and should be the same as
        the case-folded model name.  This is suitable return value for a
        ListView.get_queryset()
        """
        if not app_labels:
            app_labels = self.app_list.split(',')

        data = []
        for i in app_labels:
            data += [
                dict(app=i, table=j[len(i) + 1:])
                for j in self.get_table_names(i)
            ]
        return data

    def get_table_data(self, user_app, user_table_name):
        """
        Return table content for given app and short table name

        The parameters are untrusted, assumed to be passed from a URL and will
        be checked here.
        """
        # verify app and table name
        for i in self.get_table_names(user_app):
            if i == user_app + '_' + user_table_name:
                table_name = i
                break
        else:
            raise LookupError(
                f'no such table in snapshot: {user_app} / {user_table_name}'
            )

        rows, descr = self.do_sql('select * from ' + table_name, descr=True)

        columns = [i[0] for i in descr]
        return columns, rows

    def get_absolute_url(self):
        return reverse('snapshot', kwargs=dict(name=self.name))


class Model(models.Model):
    """
    Adds some extras to Django's Model
    """

    """
    List of str, indicating missing data as used externally.  Internally,
    None or the empty string remains in use for missing data.
    The first element is used for export.
    """
    MISSING_DATA = ['-']

    # replace the default auto field that Django adds
    id = AutoField(primary_key=True)
    history = models.ManyToManyField(ChangeRecord)

    class Meta:
        abstract = True

    objects = Manager()
    curated = CurationManager()

    average_by = ()
    """ average_by is a list of lists (or tuples) of field names over which
    taking averages makes sense """

    average_fields = []
    """ List of field names of fields other than DecimalField over which an
    average can be calculated"""

    hidden_fields = []
    """ Fields that are not displayed or exported by default, hidden from
    public view """

    field_types = {
        'numeric': (
            'AutoField',
            'BigAutoField',
            'DateField',
            'DateTimeField',
            'DecimalField',
            'DurationField',
            'FloatField',
            'IntegerField',
            'BigIntegerField',
            'PositiveIntegerField',
            'PositiveSmallIntegerField',
            'SmallIntegerField',
            'TimeField',
            'BinaryField',
        ),
        'boolean': ('BooleanField',),
        'relation': ('ForeignKey', 'OneToOneField'),
    }

    @classmethod
    def _field_type_check(cls, field, kind):
        """
        Return True if field is of the given kind

        Supported kind are the keys of the field_type class member dictionary.
        The related per-kind methods that call this can be used to check if a
        field supports certain operations.

        :param field: Either a field object or the name of a field.

        May raise ValueError if the given field does not belong to this Model
        or ???
        """
        if isinstance(field, str):
            field = cls.get_field(field)
        else:
            flist = cls.get_fields(with_m2m=True, with_reverse=True).fields
            if field not in flist:
                raise ValueError('field does not belong to model or is not '
                                 f'supported by this method: {field}')

        return field.get_internal_type() in cls.field_types[kind]

    @classmethod
    def is_numeric_field(cls, field):
        """
        Check if given field is of some numeric data type
        """
        return cls._field_type_check(field, 'numeric')

    @classmethod
    def is_bool_field(cls, field):
        """
        Check if given field is of some boolean type
        """
        return cls._field_type_check(field, 'boolean')

    @classmethod
    def is_relation_field(cls, field):
        """
        Check if given field is of a forward relation
        """
        return cls._field_type_check(field, 'relation')

    @classmethod
    def is_simple_field(cls, field):
        """
        Check if given field is "simple"

        Simple fields are not ManyToMany or ManyToOne but can be represented
        in a table cell
        """
        if isinstance(field, (models.ManyToOneRel, models.ManyToManyField)):
            return False
        else:
            return True

    @classmethod
    @cache
    def get_fields(
        cls,
        skip_auto=False,
        with_m2m=False,
        with_hidden=False,
        with_reverse=False,
        **filters,
    ):
        """
        Get fields to be displayed in table (in order) and used for import

        Should be overwritten by models as needed to include e.g. m2m fields
        Many-to-many fields are by default excluded because of the difficulties
        of meaningfully displaying them

        :param bool with_reverse: Include one_to_many relations, that is,
                                  foreign keys that point to us.
        :param bool filters: Filter by boolean field attributes.  Only return
                             fields for which the given attribute evaluates to
                             True or False respectively.  Beware interactions
                             with the other parameters and other built-in
                             filtering.
        """
        # exclude a field if test comes back True
        tests = [
            # list default tests here
            lambda x: x.name == 'history',
        ]

        if skip_auto:
            tests.append(lambda x: isinstance(x, models.AutoField))

        if not with_m2m:
            tests.append(lambda x: x.many_to_many)

        if not with_hidden:
            tests.append(lambda x: x.name in cls.hidden_fields)

        if not with_reverse:
            tests.append(lambda x: x.one_to_many)

        for k, v in filters.items():
            if not hasattr(models.Field, k):
                raise AttributeError('Only field attributes can be put in '
                                     'filters parameter')
            tests.append(lambda x: bool(getattr(x, k, None)) != bool(v))

        fields = []
        for i in cls._meta.get_fields():
            for t in tests:
                if t(i):
                    break
            else:
                fields.append(i)

        names = [i.name for i in fields]
        verbose = [getattr(i, 'verbose_name', i.name) for i in fields]
        return Fields(fields=fields, names=names, verbose=verbose)

    @classmethod
    def get_related_objects(cls):
        """
        Get compatible one-to-many related objects

        This returns Model._meta.related_objects whose related Model inherits
        from mibios.Model. These can meaningful participant in e.g. count
        columns.
        """
        return [
            i for i in cls.get_fields(with_m2m=True).fields
            if i.many_to_many
        ] + [
            i for i in cls._meta.related_objects
            if issubclass(i.related_model, Model)
            and i.one_to_many  # prevents m2ms from being returned twice
        ]

    @classmethod
    def get_related_accessors(cls):
        """
        Discover simple local and forward-looking remotely related fields

        This includes one-to-one and excludes many-to-many and one-to-many
        (reversed foreign keys) relations.  Cycles get broken at the first
        repetition of model and field name combination.
        """
        data = [('', cls, [])]
        while True:
            for pos, item in enumerate(data):
                if isinstance(item, tuple):
                    # get first tuple item
                    break
            else:
                # all done
                break

            path, model, seen = data.pop(pos)
            if path:
                pref = path + '__'
            else:
                pref = ''

            fields_s = model.get_fields(skip_auto=False)

            if pref:
                # Add item for model itself:
                # this is either pref without __ suffix or natural
                # TODO: str.removesuffix() available with Python 3.9
                data.insert(pos, pref[:-2])
                pos += 1

            if 'name' not in fields_s.names:
                if hasattr(model, 'name'):
                    # add item for name property
                    data.insert(pos, pref + 'name')
                    pos += 1
                elif not pref:
                    # for top-level and no name add natural
                    data.insert(pos, 'natural')
                    pos += 1

            for i in fields_s.fields:
                new_path = pref + i.name
                if i.many_to_one or i.one_to_one:
                    # foreign key rel
                    this = (i.related_model, i.name)
                    if this in seen:
                        # avoid cycle
                        continue
                    else:
                        new_item = (new_path, i.related_model, seen + [this])
                else:
                    new_item = new_path
                data.insert(pos, new_item)
                pos += 1

        return data

    @classmethod
    def get_related_fields(cls, relations_last=True, auto_fields=False):
        """
        Discover simple local and forward-looking remotely related fields

        This includes one-to-one and excludes many-to-many and one-to-many
        (reversed foreign keys) relations.  Cycles get broken at the first
        repetition of model and field combination.

        bool relations_last:
            If true, sort relations last, otherwise keep order of fields in
            class declaration
        """
        data = [([], cls, [])]
        while True:
            for pos, item in enumerate(data):
                if isinstance(item, tuple):
                    # get first tuple item
                    break
            else:
                # all done
                break

            path, model, seen = data.pop(pos)

            fields = model._meta.get_fields()
            if relations_last:
                fields = sorted(fields, key=lambda x: x.is_relation)
            for i in fields:

                if isinstance(i, models.AutoField) and not auto_fields:
                    continue
                if i.many_to_many:
                    continue

                new_path = path + [i]

                if i.is_relation:
                    rel1 = (i.related_model, i)
                    if i.one_to_one:
                        # also add the reverse relation
                        if hasattr(i, 'remote_field'):
                            rel2 = (i.model, i.remote_field)
                        elif hasattr(i, 'field'):
                            rel2 = (i.related_model, i.field)
                        else:
                            raise RuntimeError('this should not be reachable')
                    else:
                        rel2 = None

                    if rel1 in seen or rel2 in seen:
                        continue

                    seen.append(rel1)
                    if rel2:
                        seen.append(rel2)
                    new_item = (new_path, i.related_model, seen)

                else:
                    # normal field
                    new_item = new_path

                data.insert(pos, new_item)
                pos += 1

        return data

    @classmethod
    def get_related_accessors2(cls, relations_last=True, auto_fields=False):
        """
        List accessors to (related) fields

        This is a simpler version and potential replacement for
        get_related_accessors().  It does not specially handle id/name/natural
        fields.
        """
        fields = cls.get_related_fields(
            relations_last=relations_last,
            auto_fields=auto_fields,
        )
        return ['__'.join([i.name for i in path]) for path in fields]

    @classmethod
    def get_field(cls, accessor):
        """
        Retrieve a field object following relations

        Raises LookupError if field can not be accessed.
        """
        first, _, rest = accessor.partition('__')

        if first == 'pk':
            # tolerate pk as synonym of id
            first = 'id'

        try:
            field = cls._meta.get_field(first)
        except FieldDoesNotExist as e:
            raise LookupError(e) from e

        if rest:
            if field.related_model is None:
                raise LookupError('is not a relation field: {}'.format(field))
            else:
                return field.related_model.get_field(rest)
        else:
            return field

    @classmethod
    def get_average_fields(cls):
        """
        Get fields for which we may want to calculate averages

        Usually these are all the decimal fields
        """
        numeric_types = (
            models.DecimalField,
            models.FloatField,
            models.BooleanField,
        )
        ret = [cls.get_field(i) for i in cls.average_fields]
        for i in cls.get_fields().fields:
            if isinstance(i, numeric_types):
                if i not in ret:
                    ret.append(i)
        return ret

    def export(self):
        """
        Convert object into "table row" / list
        """
        ret = []
        for i in self._meta.get_fields():
            if self.is_simple_field(i):
                ret.append(getattr(self, i.name, None))
        return ret

    def export_dict(self, to_many=False):
        ret = OrderedDict()
        for i in self._meta.get_fields():
            if self.is_simple_field(i) or to_many:
                ret[i.name] = getattr(self, i.name, None)
        return ret

    def compare(self, other):
        """
        Compares two objects and relates them by field content

        Can be used to determine if <self> can be updated by <other> in a
        purely additive, i.e. without changing existing data, just filling
        blank fields.  <other> can also be a dict.

        Returns a tuple (bool, int), the first component of which says if both
        objects are consistent with each other, i.e. if the only differences on
        fields allowed if our's is blank and the other's is not blank.
        Differences on many-to-many fields don't affect consistency.  The
        second component contains the names of those fields that are null or
        blank in <self> but not in <other> including additional many-to-many
        links.

        For two inconsistent objects the return value's second component is
        undefined (it may be usable for debugging.)
        """
        if isinstance(other, Model):
            if self._meta.concrete_model != other._meta.concrete_model:
                return (False, None)
        elif not isinstance(other, dict):
            raise TypeError('can\'t compare to {} object'.format(type(other)))

        BLANKS = ['', None]
        is_consistent = True
        only_us = []
        only_them = []
        mismatch = []
        for i in self._meta.get_fields():
            if isinstance(other, dict) and i.name not in other:
                # other is silent on this field (dict version)
                continue

            if isinstance(i, models.ManyToOneRel):
                # a ForeignKey in third model pointing to us
                # ignore - must be handled from third model
                pass
            elif isinstance(i, models.ManyToManyField):
                ours = set(getattr(self, i.name).all())
                if isinstance(other, dict):
                    # TODO / FIXME: how do we get here?
                    try:
                        theirs = set(other['name'])
                    except TypeError:
                        # packaged in iterable for set()
                        theirs = set([other['name']])
                    except KeyError:
                        theirs = set()
                else:
                    theirs = set(getattr(other, i.name).all())
                if theirs - ours:
                    only_them.append(i.name)
            elif isinstance(i, models.OneToOneField):
                raise NotImplementedError()
            else:
                # ForeignKey or normal scalar field
                # Assumes that None and '' are not both possible values and
                # that either of them indicates missing data
                ours = getattr(self, i.name)
                if isinstance(other, dict):
                    theirs = other[i.name]
                else:
                    theirs = getattr(other, i.name)

                if ours in BLANKS and theirs in BLANKS:
                    continue

                if ours not in BLANKS and theirs not in BLANKS:
                    # both are real data
                    # usually other dict has str values
                    # try to cast to e.g. Decimal, ... (crossing fingers?)
                    if isinstance(theirs, str):
                        theirs = type(ours)(theirs)

                    if ours != theirs:
                        is_consistent = False
                        mismatch.append(i.name)
                elif ours in BLANKS:
                    # other has more data
                    only_them.append(i.name)
                else:
                    # other has data missing
                    is_consistent = False
                    only_us.append(i.name)

        diffs = dict(only_us=only_us, only_them=only_them, mismatch=mismatch)
        return (is_consistent, diffs)

    @property
    def natural(self):
        """
        A natural identifier under which the object is commonly known

        This defaults to the name field if it exists.  Models without a name
        field should implement this.

        The natural value must be derived from the non-relational fields of
        the model.  To implement the natural proterty for a model this method
        as well as natural_lookup() must be implemented.  The setter method
        should be general enough for most cases.  The inverse of this method is
        implemented by natural_lookup().
        """
        return getattr(self, 'name', self.pk)

    def natural_key(self):
        return self.natural

    @classmethod
    def natural_lookup(cls, value):
        """
        Generate a dict lookup from the natural value / key

        Used to replace a natural lookup with real lookups that Django can
        understand.  This method should be overwritten to suit the model.  The
        default implementation assumes the model has a "name" field.
        Implementations usually de-construct the natural value into its
        components and is the inverse of the natural() property.

        Might also be used as parsing tool to coerce a user-given input value
        to field-compatible values.
        """
        if value is cls.NOT_A_VALUE:
            value = None
        try:
            cls._meta.get_field('name')
        except FieldDoesNotExist:
            return dict(pk=value)
        else:
            return dict(name=value)

    @natural.setter
    def natural(self, value):
        """
        Update model fields from natural value

        The default implementation should be general enough to be used by
        inheriting classes

        Raises a Model.DoesNotExist if a foreign key lookup is used and the
        related object does not already exist.
        """
        for k, v in self.natural_lookup(value).items():
            if '__' in k:
                # set foreign key field to related instance
                field, _, lookup = k.partition('__')
                rel_model = self.get_field(field).related_model
                kw = {lookup: v}
                try:
                    k, v = field, rel_model.objects.get(**kw)
                except rel_model.DoesNotExist as e:
                    extra = f'failed on: {lookup} = {v}'
                    e.args = (*e.args, extra)
                    raise

            setattr(self, k, v)

    def __str__(self):
        ret = self.natural
        if isinstance(ret, str):
            return ret
        else:
            # is just pk
            return super().__str__()

    NOT_A_VALUE = object()

    @classmethod
    def resolve_natural_lookups(cls, *accessors, **lookups):
        """
        Detect and convert natural object lookups

        With the **lookup parameter given will convert "*__natural='foo'" and
        "*__model_name='foo'" into their proper lookups.  With the *accessors
        containing just lookup LHSs, a list of resolved accessors is returned.
        """
        if accessors and lookups:
            raise ValueError('Either *accessors or **lookup parameters can '
                             'passed but not both')

        lookups.update({i: cls.NOT_A_VALUE for i in accessors})

        ret = {}
        for lhs, rhs in lookups.items():
            if isinstance(rhs, (str, int)) or rhs is cls.NOT_A_VALUE:
                # any natural lookup should be str (or int for pk)
                # TODO: it might be useful to have natural=None cases
                # also get resolved.
                pass
            else:
                # if it's not, then the resulting error will be probably be
                # misleading
                # TODO: should something__natural=None return something=None?
                ret[lhs] = rhs
                continue

            cur_model = cls

            parts = lhs.split('__')
            for part in parts:
                if part == 'natural':
                    continue
                fields = {i.name: i for i in cur_model._meta.get_fields()}
                if part in fields and fields[part].is_relation:
                    cur_model = fields[part].related_model
                else:
                    # not an obj lookup, keep as-is
                    ret[lhs] = rhs
                    break
            else:
                # prepare lhs prefix for the natural replacements:
                if part == 'natural':
                    # remove __natural from lhs
                    lhs = '__'.join(parts[:-1])

                if lhs:
                    lhs += '__'

                if isinstance(rhs, int):
                    ret.update({lhs + 'pk': rhs})
                else:
                    try:
                        real_lookups = cur_model.natural_lookup(rhs)
                    except Exception as e:
                        # Assume code in natural_lookup() is correct and
                        # treat this a user error, i.e. the natural rhs is
                        # bad
                        msg = 'Failed to resolve: [{}]{}={}' \
                              ''.format(cur_model._meta.model_name, parts, rhs)
                        raise NaturalKeyLookupError(msg) from e

                    ret.update({lhs + k: v for k, v in real_lookups.items()})
        if accessors:
            return list(ret.keys())
        else:
            return ret

    @classmethod
    def str_blank(cls, *values):
        """
        Convert into strings, explicitly marking blank/null values as such

        Use this when an empty string is insufficient to indicate missing data.
        This is the reverse of decode_blank().
        """
        ret = []
        for i in values:
            if i in ['', None]:
                ret.append(cls.MISSING_DATA[0])
            else:
                ret.append(str(i))
        return ret[0] if len(ret) == 1 else tuple(ret)

    @classmethod
    def decode_blank(cls, *values):
        """
        Filter values, turning strings indicating missing data into actual
        blank/empty string.
        """
        ret = []
        for i in values:
            if i in cls.MISSING_DATA:
                ret.append('')
            else:
                ret.append(i)
        return ret[0] if len(ret) == 1 else tuple(ret)

    def get_absolute_url(self):
        name = 'admin:{app}_{model}_change' \
               ''.format(app=self._meta.app_label, model=self._meta.model_name)
        return reverse(name, kwargs=dict(object_id=self.pk))

    def add_change_record(self, is_created=None, is_deleted=False, file=None,
                          line=None, user=None, comment=''):
        """
        Create a change record attribute for this object

        If the object has no id/pk yet the change will be "is_created".  The
        fields will remain empty until save()

        Call this before the objects save() or delete() method.
        """
        self.change = ChangeRecord(
            user=user,
            file=file,
            line=line,
            comment=comment,
            record=self,
            record_natural=self.natural,  # None for new but not yet saved objs
            is_created=is_created or (self.id is None),
            is_deleted=is_deleted,
        )

    @transaction.atomic
    def save(self, *args, **kwargs):
        is_created = self.id is None
        super().save(*args, **kwargs)

        if self.history is None:
            return

        if not hasattr(self, 'change'):
            self.add_change_record(is_created=is_created)
        if self.change.record_natural is None:
            # natural may still be None if record is new and change was created
            # manually and the model uses the fallback to pk for natural
            # property but now after save() we have a valid pk
            self.change.record_natural = self.natural

        # set record (again) as super().save() resets this to None for unknown
        # reasons:
        self.change.record = self
        if self.change.has_changed():
            self.change.save()

    def delete(self, *args, **kwargs):
        if self.history is None:
            return super().delete(*args, **kwargs)
        else:
            if not hasattr(self, 'change'):
                self.add_change_record(is_deleted=True)
            with transaction.atomic():
                self.change.save()
                return super().delete(*args, **kwargs)

    def full_clean(self, *args, **kwargs):
        """
        Validate the object

        Add model name to super()'s error dict
        """
        try:
            super().full_clean(*args, **kwargs)
        except ValidationError as e:
            errors = e.update_error_dict({
                'model_name': self._meta.model_name,
            })
            raise ValidationError(errors) from e

    @classmethod
    def get_serializer_class(self):
        """
        Return REST API Serializer class
        """
        fields = ['url'] + self.get_fields().names
        for i in self._meta.related_objects:
            fields.append(i.name + '_set')

        meta_opts = dict(model=self._meta.model, fields=fields)
        Meta = type('Meta', (object,), meta_opts)
        name = self._meta.model_name.capitalize() + 'Serializer'
        opts = dict(Meta=Meta)
        return type(name, (HyperlinkedModelSerializer,), opts)

    @classmethod
    def get_rest_api_viewset_class(self):
        """
        Return REST framework ViewSet class
        """
        opts = dict(
            queryset=self.curated.all(),
            serializer_class=self.get_serializer_class(),
        )
        name = self._meta.model_name.capitalize() + 'RESTViewSet'
        return type(name, (ReadOnlyModelViewSet,), opts)

    def get_value_related(self, lookup):
        """
        Retrieve value, possibly following relations

        Like QuerySet.values() but for single instance.
        """
        name, _, rest = lookup.partition('__')
        val = getattr(self, name)
        if rest:
            return val.get_value_related(rest)
        else:
            return val

    def getter(self, *accessors, raise_on_error=False):
        """
        Return possibly related field values

        :param str accessors: Accessors with __ lookup divider.
        :param bool raise_on_error: If True and a related field is
                                    inaccessible because e.g. the related
                                    object is None, then raise an
                                    AttributeError.  If False, will return None
                                    in these cases

        Comparable to operator.attrgetter(), but returns a tuple of the same
        length as accessors.
        """
        vals = []
        for a in accessors:
            obj = self
            for i in a.split('__'):
                try:
                    obj = getattr(obj, i)
                except AttributeError:
                    if raise_on_error:
                        raise
                    obj = None
            vals.append(obj)
        return tuple(vals)


class ParentModel(Model):
    """
    Class from which parent models can be derived
    """
    class Meta:
        abstract = True

    @classmethod
    def get_child_info(cls):
        """
        Get a mapping from inheriting models to their relation field
        """
        info = {}
        for field in cls._meta.get_fields():
            if field.one_to_one:
                for child in cls.__subclasses__():
                    if child == field.related_model:
                        info[child] = field
        return info

    def get_child_relation(self):
        """
        Get the relation to the child
        """
        for i in self.get_child_info().values():
            try:
                getattr(self, i.name)
            except i.related_model.DoesNotExist:
                pass
            else:
                return i
        return LookupError('no child relation')

    @property
    def child(self):
        """
        Polymorphic field pointing to the child instance
        """
        return getattr(self, self.get_child_relation().name)

    @property
    def natural(self):
        """
        Get natural property from child instance
        """
        return self.child.natural

    @classmethod
    def natural_lookup(cls, value):
        errors = []
        for model_class, field in cls.get_child_info().items():
            try:
                kwargs = model_class.natural_lookup(value)
            except Exception as e:
                errors.append((model_class, e.__class__.__name__, e))
            else:
                # add with correct relation name added
                return {field.name + '__' + k: v for k, v in kwargs.items()}
        else:
            raise NaturalKeyLookupError(errors)


class TagNote(Model):
    tag = models.CharField(max_length=100, default='info', db_index=True)
    name = models.CharField(max_length=100, unique=True)
    text = models.TextField(max_length=5000, blank=True)

    class Meta:
        verbose_name = 'tags and notes'
        verbose_name_plural = verbose_name
