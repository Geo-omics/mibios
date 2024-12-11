from datetime import datetime
from itertools import chain
from logging import getLogger
import re

from django.apps import apps
from django.conf import settings
from django.contrib.auth.models import Group
from django.contrib.postgres.search import SearchQuery
from django.core.exceptions import FieldDoesNotExist
from django.db import connections, transaction, NotSupportedError
from django.db.models.signals import post_save
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.html import escape as escape_html
from django.utils.module_loading import import_string

from mibios.omics.models import AbstractSample
from mibios.omics.managers import SampleLoader as OmicsSampleLoader
from mibios.umrad.manager import (
    InputFileError, Loader, MetaDataLoader, Manager,
)
from mibios.umrad.model_utils import delete_all_objects_quickly
from mibios.umrad.utils import CSV_Spec, atomic_dry, SkipRow, SpecError
from mibios.omics.models import SampleTracking
from mibios.omics.utils import get_sample_blocklist

from .search_fields import search_fields


log = getLogger(__name__)


class BoolColMixin:
    def parse_bool(self, value, obj):
        """ Pre-processor to parse booleans """
        # Only parse str values.  The pandas reader may give us booleans
        # already for some reason (for the modified_or_experimental but not the
        # has_paired data) ?!?
        if isinstance(value, str) and value:
            if value.casefold() == 'false':
                return False
            elif value.casefold() == 'true':
                return True
            else:
                raise InputFileError(
                    f'expected TRUE or FALSE (any case) but got: {value}'
                )
        return value


class AboutInfoManager(Manager):
    @atomic_dry
    def new(self, copy_last=True):
        """ Create and save a new instance """
        parent = self.order_by('pk').last()
        if parent is not None and parent.when_published is None:
            raise RuntimeError(
                'An unpublished entry still exists.  Update it or publish it '
                'first!'
            )

        obj = self.model()
        if copy_last and parent is not None:
            obj.generation = parent.generation + 1
        else:
            obj.generation = 0

        obj.auto_update()
        obj.full_clean()
        obj.save()
        if copy_last and parent is not None:
            obj.credits.add(*parent.credits.all())
        return obj

    @atomic_dry
    def auto_update(self):
        """ Auto-update certain fields of the last unpublished entry """
        last = self.order_by('pk').last()
        if last is None:
            raise self.model.DoesNotExist
        if last.when_published is not None:
            raise RuntimeError('last entry is already published')
        last.auto_update()
        last.save()
        return last

    @atomic_dry
    def publish(self):
        """ publish the latest entry """
        last = self.last()
        if last is None:
            raise self.model.DoesNotExist
        if last.when_published is not None:
            raise RuntimeError(f'is already published: {vars(last)=}')
        last.when_published = timezone.localtime()
        last.save()
        return last


class DatasetLoader(BoolColMixin, MetaDataLoader):
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A', 'TBD']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - studies_datasets.tsv'

    def ensure_id(self, value, obj):
        """ Pre-processor to skip rows without dataset id """
        if not value:
            raise SkipRow('no dataset id')

        return value

    def check_status(self, value, obj):
        match value:
            case 'Completed' | 'In Progress' | 'Issues':
                return self.spec.IGNORE_COLUMN
            case _:
                self.skipped.append(obj)
                raise SkipRow(f'study status: {obj.dataset_id} not ready yet')

    def check_is_glamr(self, value, obj):
        if self.parse_bool(value, obj):
            return self.spec.IGNORE_COLUMN
        else:
            self.skipped.append(obj)
            raise SkipRow(f'{obj.dataset_id} is not a GLAMR dataset')

    def sub_user_groups(self, value, obj):
        """
        Parse special values in the 'private' column

        Substitute the staff group name if appropriate.  Public access datasets
        must be marked as such, we return an empty list (blank) then.  An
        incoming blank however is interpreted as staff-only, just to be safe.
        Otherwise we expect a list of group names, though usually just a single
        name.
        """
        # reserved names, case-folded:
        STAFF_VAL = ['true', 'private', 'glamr-staff', 'glamr_staff', 'staff']
        PUBLIC_VAL = ['false', 'public']
        RESERVED = STAFF_VAL + PUBLIC_VAL

        if value is None:
            return value
        if value.casefold() in PUBLIC_VAL:
            # public access
            return None
        elif not value or value.casefold() in STAFF_VAL:
            # private data
            return ((self.staff_group_name,),)
        else:
            group_names = self.split_m2m_value(value)
            for name in group_names:
                if name.casefold() in RESERVED:
                    # The reserved names must be alone
                    raise SkipRow(f'reserved value "{name}" in group listing')
                # Since Group.objects is just the normal Django-supplied
                # manager, new groups won't be auto-created during loading, so
                # we create them here and now
                try:
                    grp = Group.objects.filter(name__iexact=name).get()
                except Group.DoesNotExist:
                    Group.objects.create(name=name)
                    print(f'New user group created: {name}')
                else:
                    if name != grp.name:
                        # found a similar group, name differs in case only
                        raise SkipRow(f'similar group name exist: value '
                                      f'"{value}" vs. group {grp}')

            # inject the staff group, gets access to all non-public datasets
            return tuple(
                [(self.staff_group_name, )]
                + [(i, ) for i in group_names]
            )

    def split_by_comma(self, value, obj):
        return [(i, ) for i in self.split_m2m_value(value, sep=',')]

    spec = CSV_Spec(
        ('dataset', 'dataset_id', ensure_id),  # A
        ('Associated_papers', 'references', split_by_comma),  # B
        ('Primary_pub', 'primary_ref.reference_id'),  # C
        ('primary_pub_title', None),  # D ignore
        ('NCBI_BioProject', 'bioproject'),  # E
        # other_ncbi_ids  # F  TODO: add this?
        ('JGI_Project_ID', 'jgi_project'),  # G
        ('GOLD_ID', 'gold_id'),  # H
        ('MG-RAST_study', 'mgrast_study'),  # I
        ('Location and Sampling Scheme', 'scheme'),  # J
        ('Material Type', 'material_type'),  # L
        ('Water Bodies', 'water_bodies'),  # M
        ('Primers', 'primers'),  # N
        ('Sequencing targets', 'sequencing_target'),  # O
        ('Sequencing Platform', 'sequencing_platform'),  # P
        ('Size Fraction(s)', 'size_fraction'),  # Q
        ('study_status', None, check_status),  # R
        # sample_added_by  # S ignore
        ('private', 'restricted_to', sub_user_groups),  # T
        ('is_GLAMR', None, check_is_glamr),  # U
        ('Notes', 'note'),  # V
        # ignoring counts columns
        extra={'restricted_to': {
            'acc_field': Group._meta.get_field('name'),
            'acc_lookup': 'name',
        }},
    )

    def load(self, *args, **kwargs):
        """
        Load Dataset records from the datasets/studies google sheet
        """
        # avoid circular import:
        self.staff_group_name = import_string('mibios.glamr.accounts.STAFF_GROUP_NAME')  # noqa:E501
        self.skipped = []
        super().load(*args, **kwargs)
        if connections['default'].vendor == 'postgresql':
            self.model.objects.update_access()
        skipped_ids = [getattr(i, 'dataset_id', None) for i in self.skipped]
        qs = self.filter(dataset_id__in=skipped_ids)
        if qs.exists():
            print('WARNING: These datasets exist in the DB but got skipped:')
            for i in qs:
                print('    ', i.dataset_id)


class ReferenceLoader(MetaDataLoader):
    empty_values = ['NA', 'Not Listed']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - papers.tsv'

    def fix_doi(self, value, obj):
        """ Pre-processor to fix issue with some DOIs """
        if value is not None:
            # fix, don't require umich weblogin to follow these links
            value = value.replace('doi-org.proxy.lib.umich.edu', 'doi.org')
        return value

    def check_empty(self, value, obj):
        """ Pre-processor to determine if row needs to be skipped """
        if not value:
            raise SkipRow('field is empty')

        return value

    spec = CSV_Spec(
        ('PaperID', 'reference_id', check_empty),
        ('Reference', 'short_reference', check_empty),
        ('pub_year', 'year'),
        ('Authors', 'authors'),
        ('last_author', 'last_author'),
        ('Title', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi', fix_doi),
        # ignoring status, notes, entry priority columns
    )


class SampleInputSpec(CSV_Spec):
    UNITS_SHEET = 'Great_Lakes_Omics_Datasets.xlsx - metadata_units_and_notes.tsv'  # noqa: E501

    def setup(self, loader, column_specs=None, **kwargs):
        self.has_header = True
        if column_specs is None:
            base_spec0 = list(self._spec)
        else:
            base_spec0 = column_specs

        # base_spec is assumed to be in three-column (col_name, field_name,
        # prep-function) format
        base_spec = {}
        for col_name, field_name, *preps in base_spec0:
            if field_name in base_spec:
                raise SpecError(
                    f'field name duplicate: {field_name=} {base_spec0=}'
                )
            base_spec[field_name] = (col_name, field_name, *preps)

        specs = []
        with (settings.GLAMR_META_ROOT / self.UNITS_SHEET).open() as ifile:
            # First: what columns do we want?
            header = ifile.readline().rstrip('\n').split('\t')
            col_name_col = header.index('Column_name')
            field_name_col = header.index('django_field_name')
            for line in ifile:
                row = line.rstrip('\n').split('\t')
                col_name = row[col_name_col]
                field_name = row[field_name_col]
                if not field_name:
                    if col_name in base_spec:
                        raise RuntimeError('field name missing in meta data')
                    # row does not relate to a existing field
                    continue
                if field_name in base_spec:
                    if base_spec[field_name][0] != col_name:
                        raise RuntimeError(
                            f'fieldname mismatch for "{col_name}": '
                            f'{base_spec[field_name][0]=} != {col_name=}\n'
                            f'{line=}'
                        )
                    # override UNITS_SHEET data with base spec item
                    spec_item = base_spec.pop(field_name)
                else:
                    # take from meta data
                    spec_item = (col_name, field_name)

                specs.append(spec_item)

        # concatenate with remaining base specs
        specs = list(base_spec.values()) + specs

        # Check that the spec account for all field we think should get loaded
        # from the file:
        fields_accounted_for = set((i[1] for i in specs))
        required = [  # fields from AbstractSample that we want to check for
            'sample_id', 'sample_name', 'sample_type', 'has_paired_data',
            'sra_accession', 'amplicon_target', 'fwd_primer', 'rev_primer',
        ]
        for i in AbstractSample._meta.get_fields():
            if i.name not in required:
                fields_accounted_for.add(i.name)
        for i in loader.model._meta.get_fields():
            if i.concrete and i.name not in fields_accounted_for:
                print(f'[WARNING] field missing from spec: {i}')

        super().setup(loader, column_specs=specs, **kwargs)


class SampleLoader(BoolColMixin, OmicsSampleLoader):
    """ loader for Great_Lakes_Omics_Datasets.xlsx """
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A', 'ND', 'not applicable',
                    'N/A']

    def load_all_meta_data(self, dry_run=False):
        """
        Convenience method -- load/update all meta data

        Loads all reference/dataset/sample data
        load meta data assuming empty DB -- reference implementation
        """
        Reference = import_string('mibios.glamr.models.Reference')
        Dataset = import_string('mibios.glamr.models.Dataset')
        with transaction.atomic():
            Reference.loader.load()
            Dataset.loader.load()
            self.load_meta()
            self.update_from_pipeline_registry(quiet=True, skip_on_error=True)
            if dry_run:
                transaction.set_rollback(True)

    def get_file(self):
        return settings.GLAMR_META_ROOT / 'Great_Lakes_Omics_Datasets.xlsx - samples.tsv'  # noqa:E501

    def fix_sample_id(self, value, obj):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    sample_id_pat = re.compile(r'^samp_[0-9]+$')

    def check_sample_id(self, value, obj):
        """
        allow skipping row based on blank or blocked sample_id

        Also checks the whole row against the blocklist.
        """
        if not self.sample_id_pat.match(value):
            # skip row for invalid sample id
            raise SkipRow('invalid sample_id')

        if value in self.blocklist:
            if blocked_fields := self.blocklist[value]:
                # mask values for fields on blocklist
                row_data = enumerate(self.current_row_data[1:], start=1)
                for pos, (field, fn, _) in row_data:
                    if field.name in blocked_fields:
                        self.current_row_data[pos] = \
                                (field, fn, self.spec.IGNORE_COLUMN)
            else:
                # no fields specified means block whole sample
                raise SkipRow('found in blocklist')

        # check other values in row
        for f, _, other in self.current_row_data[1:]:
            if other in self.empty_values or other is None or other is self.spec.IGNORE_COLUMN:  # noqa:E501
                continue
            else:
                break  # row is non-empty
        else:
            # consider row blank
            raise SkipRow('row seems empty, except sample id')

        # keep going normally
        return value

    # re for yyyy or yyyy-mm (year or year/month only) timestamp formats
    partial_date_pat = re.compile(r'^([0-9]{4})(?:-([0-9]{2})?)$')

    def process_timestamp(self, value, obj):
        """
        Pre-processor for the collection timestamp

        1. Partial dates: years only get saved as Jan 1st and year-month gets
           the first of month; this is indicated by the collection_ts_partial
           field, which is set here.
        2. Add timezone to timestamps if needed (and hoping the configued TZ is
           appropriate).  This avoids spamming of warnings from the
           DateTimeField's to_python() method in some cases.
        """
        if value is None:
            return None

        # step 0. temp fix for some input
        value = value.removesuffix('TNA')

        orig_value = value
        del value

        try:
            value = parse_date(orig_value)
        except ValueError as e:
            raise InputFileError(f'is date format but invalid: {e}') from e

        if value is None:
            # not a date, try datetime
            try:
                value = parse_datetime(orig_value)
            except ValueError as e:
                raise InputFileError(
                    f'is datetime format but invalid: {e}'
                ) from e

            # got a complete timestamp
            collection_ts_partial = self.model.FULL_TIMESTAMP
        else:
            # got a date, add fake time (midnight)
            value = datetime(value.year, value.month, value.day)
            collection_ts_partial = self.model.DATE_ONLY

        if value is None:
            # failed parsing as ISO 8601, try for year-month and year only
            # which MIMARKS/MIXS allows
            m = self.partial_date_pat.match(orig_value)

            if m is None:
                raise InputFileError(f'failed parsing timestamp: {orig_value}')
            else:
                year, month = m.groups()
                year = int(year)
                if month is None:
                    month = 1
                    collection_ts_partial = self.model.YEAR_ONLY
                else:
                    month = int(month)
                    collection_ts_partial = self.model.MONTH_ONLY

                try:
                    value = datetime(year, month, 1)
                except ValueError as e:
                    # invalid year or month
                    raise InputFileError('failed parsing timestamp', e) from e

        # value should now be a datetime instance
        if value.tzinfo is None:
            default_timezone = timezone.get_default_timezone()
            value = timezone.make_aware(value, default_timezone)

        obj.collection_ts_partial = collection_ts_partial
        return value

    def parse_human_int(self, value, obj):
        """
        Pre-processor to allow use of commas to separate thousands
        """
        # TODO: check if this is still needed of if commas were removed
        # upstream
        if value:
            return value.replace(',', '')
        else:
            return value

    spec = SampleInputSpec(
        ('SampleID', 'sample_id', check_sample_id),  # A
        # id_fixed B  --> ignore
        # sample_input_complete C  --> ignore
        ('SampleName', 'sample_name'),  # D
        ('StudyID', 'dataset.dataset_id'),  # E
        ('ProjectID', 'project_id'),  # F
        ('Biosample', 'biosample'),  # G
        ('Accession_Number', 'sra_accession'),  # H
        ('GOLD_analysis_projectID', 'gold_analysis_id'),  # I
        ('GOLD_sequencing_projectID', 'gold_seq_id'),  # J
        ('JGI_study', 'jgi_study'),  # K
        ('JGI_biosample', 'jgi_biosample'),  # L
        ('sample_type', 'sample_type'),  # M
        ('has_paired_data', 'has_paired_data', 'parse_bool'),  # N
        ('amplicon_target', 'amplicon_target'),  # O
        ('F_primer', 'fwd_primer'),  # P
        ('R_primer', 'rev_primer'),  # Q
        # columns after Q, mostly defined by units sheet
        ('collection_date', 'collection_timestamp', process_timestamp),  # V
        (CSV_Spec.CALC_VALUE, 'collection_ts_partial', None),
        ('keywords', 'keywords'),  # AA
        ('pH', 'ph'),  # AG
        ('microcystis_count', 'microcystis_count', parse_human_int),  # BL
        ('planktothrix_count', 'planktothrix_count', parse_human_int),  # BM
        ('anabaena_D_count', 'anabaena_d_count', parse_human_int),  # BN
        ('samp_collect_device', 'sampling_device'),  # BX
        ('modified_or_experimental', 'modified_or_experimental', 'parse_bool'),
        ('is_isolate', 'is_isolate', 'parse_bool'),  # BT
        ('is_blank_neg_control', 'is_neg_control', 'parse_bool'),  # BU
        ('is_mock_community_or_pos_control', 'is_pos_control', 'parse_bool'),
        ('Notes', 'notes'),  # CQ
    )

    @atomic_dry
    def load_meta(self, **kwargs):
        """ samples meta data """
        self._saved_samples = []
        post_save.connect(self.on_save, sender=self.model)
        self.blocklist = {
            key: [i for i in field_list if i != 'omics']
            for key, field_list in get_sample_blocklist().items()
        }
        self.load(**kwargs)
        if connections['default'].vendor == 'postgresql':
            self.model.objects.update_access()
        flag = SampleTracking.Flag.METADATA
        for i in self._saved_samples:
            tr, new = SampleTracking.objects.get_or_create(sample=i, flag=flag)
            if not new:
                tr.save()  # update timestamp

    def on_save(self, instance=None, **kwargs):
        """
        callback for sample post_save

        Remember samples for tracking
        """
        self._saved_samples.append(instance)


class SearchableManager(Loader):
    @atomic_dry
    def reindex(self):
        delete_all_objects_quickly(self.model)
        from .search_utils import update_spellfix
        for model in search_fields.keys():
            self.index_model(model)
        update_spellfix()

    def index_model(self, model):
        """
        update the search index
        """
        if model._meta.model_name == 'compoundname':
            abund_lookup = 'compoundrecord__abundance'
        elif model._meta.model_name == 'functionname':
            abund_lookup = 'funcrefdbentry__abundance'
        else:
            try:
                model._meta.get_field('abundance')
            except FieldDoesNotExist:
                abund_lookup = None
            else:
                abund_lookup = 'abundance'
        print(f'Collecting searchable text for {model._meta.verbose_name}... ',
              end='', flush=True)
        if abund_lookup:
            # get PKs of objects with hits/abundance
            f = {abund_lookup: None}
            whits = model.objects.exclude(**f).values_list('pk', flat=True)
            whits = set(whits)
            print(f'with hits: {len(whits)} / total: ', end='', flush=True)
        else:
            whits = set()

        simple_fields = []
        related_fields = []
        for i in search_fields[model]:
            if '__' in i:
                related_fields.append(i)
            else:
                simple_fields.append(i)

        data_objs = model.objects.only('pk', *simple_fields).in_bulk()
        print(f'{len(data_objs)} [OK]')

        def get_objs_simple():
            """ generator over searchable objs - simple fields """
            for obj in data_objs.values():
                for field_name in simple_fields:
                    txt = getattr(obj, field_name)
                    if txt is None or txt == '':
                        # don't index blanks
                        continue

                    yield self.model(
                        text=escape_html(txt),
                        has_hit=(obj.pk in whits),
                        field=field_name,
                        content_object=obj,
                    )

        it = get_objs_simple()

        # The relation across which we're searching may be *-to-many, so there
        # may be multiple values per data object with multiple respective
        # Searchable objects that need to be created.  Hence, we process those
        # separately for each such field below:
        for field_name in related_fields:
            # exclude the non-related as to not index blanks
            field_name_pref = '__'.join(field_name.split('__')[:-1])
            qs = model.objects.exclude(**{field_name_pref: None})
            qs = qs.values_list('pk', field_name)
            it = chain(it, (
                self.model(
                    text=value,
                    has_hit=(pk in whits),
                    field=field_name,
                    content_object=data_objs[pk],
                )
                for pk, value in qs
                if value not in (None, '')
            ))

        self.bulk_create(it)

    def tsquery_from_str(self, query, search_type='websearch'):
        """
        Get the postgresql tsquery from the given user input

        Should returns a str, possibly empty (for legitimate reasons), or None
        if query was None.
        """
        if connections[self.db].vendor != 'postgresql':
            raise NotSupportedError('this method requires PostgreSQL')

        try:
            func = SearchQuery.SEARCH_TYPES[search_type]
        except KeyError:
            raise ValueError('invalid search_type')

        sql = f"SELECT {func}('english', %s)"
        with connections[self.db].cursor() as cur:
            cur.execute(sql, [query])
            res = cur.fetchall()

        # result is list of tuples, expect single item inside
        if len(res) == 1:
            res = res[0]
            if len(res) == 1:
                return res[0]

        raise RuntimeError(f'don\'t know how to handle query reply: {res}')

    @staticmethod
    def parse_tsquery(tsquery):
        """
        limited-depth parser for postgresql tsquery strings

        In tsquery syntax | binds most loosely and & second-most loosely.  So
        we can think of a tsquery as a disjunction of conjunctions.  Though the
        postgresql documentation on this is a bit sparse, when considering only
        the connectives | and &, it looks like the plain, phrase, and websearch
        search types do not involve queries that are further nested than this.
        It's in disjunctive normal form!

        We do not further analyse the tokens that make up the inner
        conjunctions.

        Returns a non-empty list of non-empty lists of non-empty strings.
        """
        if not isinstance(tsquery, str):
            raise TypeError('tsquery is not a str')

        if not tsquery:
            raise ValueError('tsquery is empty')

        dnf = []
        for disjunct in tsquery.split(' | '):
            clause = []
            for token in disjunct.split(' & '):
                if token:
                    clause.append(token)
                else:
                    raise ValueError(f'empty token?: {token}')
            if clause:
                dnf.append(clause)
            else:
                raise ValueError(f'empty clause?: {clause}')

        return dnf

    @classmethod
    def get_fallback_tsquery(cls, tsquery_str):
        """
        loosen up search criteria

        Takes a tsquery string and transforms it into one where multiple
        positive &-connected tokens become |-connected instead.  If they were
        also &-connected to any negative tokens, then those negative token are
        distributed to each positive token.

        This widens a search while reducing precision.
        """
        tsquery_dnf = cls.parse_tsquery(tsquery_str)

        new_dnf = []
        for clause in tsquery_dnf:
            neg = []
            pos = []
            for token in clause:
                if token.startswith('!'):
                    neg.append(token)
                else:
                    pos.append(token)
            if len(pos) >= 2:
                # split clause
                for pos_token in pos:
                    new_dnf.append(list(neg) + [pos_token])
            else:
                # do nothing
                new_dnf.append(clause)

        return ' | '.join([' & '.join(i) for i in new_dnf])


class UniqueWordManager(Loader):
    @atomic_dry
    def reindex(self):
        """
        Populate table with unique, lower-case words from Searchable.text

        Run this after running Searchable.objects.reindex().  This will not run
        on sqlite3 just on postgresql.
        """
        Searchable = apps.get_app_config('glamr').get_model('searchable')
        with connections['default'].cursor() as cur:
            table = self.model._meta.db_table
            cur.execute(f'TRUNCATE TABLE {table}')
            searchable_table = Searchable._meta.db_table
            cur.execute(
                # cf. PostgreSQL documentation F.35.5.
                f"INSERT INTO {table} (word) "
                f"SELECT word FROM "
                f"ts_stat('SELECT to_tsvector(''simple'', regexp_replace(text, ''[-~\\./\\+0-9]+'', '' '', 1, 0)) FROM "  # noqa: E501
                f"{searchable_table}')"
            )
