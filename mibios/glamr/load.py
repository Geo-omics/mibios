from datetime import datetime
import re

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import connections
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime

from mibios.omics.managers import SampleManager as OmicsSampleManager
from mibios.omics.models import AbstractSample
from mibios.umrad.manager import InputFileError, Loader, Manager
from mibios.umrad.model_utils import delete_all_objects_quickly
from mibios.umrad.utils import CSV_Spec, atomic_dry

from .search_fields import SEARCH_FIELDS


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


class MetaDataLoader(Loader):
    default_load_kwargs = dict(
        validate=True,
        bulk=False,
        update=True,
        diff_stats=True,
    )


class DatasetLoader(BoolColMixin, MetaDataLoader):
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - studies_datasets.tsv'

    def ensure_id(self, value, obj):
        """ Pre-processor to skip rows without dataset id """
        if not value:
            return self.spec.SKIP_ROW

        return value

    def get_reference_ids(self, value, obj):
        # FIXME: is unused, can this be removed?
        if value is None or value == '':
            return self.spec.IGNORE_COLUMN

        Reference = self.model._meta.get_field('reference').related_model
        id_lookups = Reference.get_accession_lookups()

        try:
            ref = Reference.objects.get(short_reference=value)
        except Reference.DoesNotExist as e:
            msg = f'unknown reference: {value}'
            raise InputFileError(msg) from e
        except Reference.MultipleObjectsReturned as e:
            # FIXME: keep this for initial dev
            msg = f'reference is not unique: {value}'
            raise InputFileError(msg) from e

        return tuple((getattr(ref, i) for i in id_lookups))

    def clean_ref_id(self, value, obj):
        """ handle some too-human temposrary field content """
        if value == 'TBD':
            return None
        return value

    spec = CSV_Spec(
        ('dataset', 'dataset_id', ensure_id),
        # ('Primary_pub', 'reference', get_reference_ids),
        ('Primary_pub', 'reference.reference_id', 'clean_ref_id'),
        ('primary_pub_title', None),
        ('NCBI_BioProject', 'bioproject'),
        ('JGI_Project_ID', 'jgi_project'),
        ('GOLD_ID', 'gold_id'),
        ('MG-RAST_study', None),  # TODO: add
        ('Location and Sampling Scheme', 'scheme'),
        ('Material Type', 'material_type'),
        ('Water Bodies', 'water_bodies'),
        ('Primers', 'primers'),
        ('Sequencing targets', 'sequencing_target'),
        ('Sequencing Platform', 'sequencing_platform'),
        ('Size Fraction(s)', 'size_fraction'),
        ('private', 'private', 'parse_bool'),
        ('Notes', 'note'),
    )


class ReferenceLoader(MetaDataLoader):
    empty_values = ['NA', 'Not Listed']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - papers.tsv'

    def fix_doi(self, value, obj):
        """ Pre-processor to fix issue with some DOIs """
        if value is not None and 'doi-org.proxy.lib.umich.edu' in value:
            # fix, don't require umich weblogin to follow these links
            value = value.replace('doi-org.proxy.lib.umich.edu', 'doi.org')
        return value

    def check_skip(self, value, obj):
        """ Pre-processor to determine if row needs to be skipped """
        if value == 'paper_17':
            return self.spec.SKIP_ROW

        if value == 'Koeppel et al. 2022':
            return self.spec.SKIP_ROW

        if not value:
            return self.spec.SKIP_ROW

        return value

    spec = CSV_Spec(
        ('PaperID', 'reference_id', check_skip),
        ('Reference', 'short_reference', check_skip),
        ('Authors', 'authors'),
        ('Title', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi', fix_doi),
        ('Associated_datasets', None),  # TODO: handle this
    )


class SampleInputSpec(CSV_Spec):
    UNITS_SHEET = 'Great_Lakes_Omics_Datasets.xlsx - metadata_units_and_notes.tsv'  # noqa: E501

    def setup(self, loader, column_specs=None, path=None):
        self.has_header = True
        if column_specs is None:
            base_spec = list(self._spec)
        else:
            base_spec = column_specs

        # base_spec is assumed to be in three-column (col_name, field_name,
        # function) format
        base_spec = {i[0]: i for i in base_spec}  # map col name to spec item

        specs = []
        with (settings.GLAMR_META_ROOT / self.UNITS_SHEET).open() as ifile:
            ifile.readline()  # ignore header
            for line in ifile:
                col_name, *_, field_name, _ = line.rstrip('\n').split('\t')
                if not field_name:
                    if col_name in base_spec:
                        raise RuntimeError('field name missing in meta data')
                    # row does not related to a existing field
                    continue
                if col_name in base_spec:
                    if base_spec[col_name][1] != field_name:
                        raise RuntimeError(f'fieldname mismatch at {col_name}')
                    # merge spec item
                    spec_item = base_spec.pop(col_name)
                else:
                    # take from meta data
                    spec_item = (col_name, field_name)

                specs.append(spec_item)

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

        super().setup(loader, column_specs=specs, path=path)


class SampleLoader(BoolColMixin, MetaDataLoader):
    """ loader for Great_Lakes_Omics_Datasets.xlsx """
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A', 'ND', 'not applicable']

    def get_file(self):
        return settings.GLAMR_META_ROOT / 'Great_Lakes_Omics_Datasets.xlsx - samples.tsv'  # noqa:E501

    def fix_sample_id(self, value, obj):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    def check_empty(self, sample_id_value, obj):
        """ check to allow skipping essentially empty rows """
        if not sample_id_value:
            return self.spec.SKIP_ROW

        # check other values in row
        for f, _, value in self.current_row_data[1:]:
            if value is None or value is self.spec.IGNORE_COLUMN:
                continue
            else:
                # keep going normally
                return sample_id_value

        # consider row blank
        return self.spec.SKIP_ROW

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

        old_value = value
        del value

        try:
            value = parse_datetime(old_value)
        except ValueError as e:
            # invalid date/time
            raise InputFileError(e) from e

        if value is None:
            # failed parsing as datetime, try date
            try:
                value = parse_date(old_value)
            except ValueError as e:
                # invalid date
                raise InputFileError(e) from e

            if value is not None:
                # add fake time (midnight)
                value = datetime(value.year, value.month, value.day)
                collection_ts_partial = self.model.DATE_ONLY
        else:
            # got a complete timestamp
            collection_ts_partial = self.model.FULL_TIMESTAMP

        if value is None:
            # failed parsing as ISO 8601, try for year-month and year only
            # which MIMARKS/MIXS allows
            m = self.partial_date_pat.match(old_value)

            if m is None:
                raise InputFileError(f'failed parsing timestamp: {old_value}')
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
        ('SampleID', 'sample_id', check_empty),  # A
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
        ('sampling_device', 'sampling_device'),  # BR
        ('modified_or_experimental', 'modified_or_experimental', 'parse_bool'),
        ('is_isolate', 'is_isolate', 'parse_bool'),  # BT
        ('is_blank_neg_control', 'is_neg_control', 'parse_bool'),  # BU
        ('is_mock_community_or_pos_control', 'is_pos_control', 'parse_bool'),
        ('modified_or_experimental', 'modified_or_experimental', 'parse_bool'),
        ('Notes', 'notes'),  # CQ
    )

    @atomic_dry
    def load_meta(self, **kwargs):
        """ samples meta data """
        template = dict(meta_data_loaded=True)
        return self.load(template=template, **kwargs)


class SearchableManager(Loader):
    @atomic_dry
    def reindex(self):
        delete_all_objects_quickly(self.model)
        models = [
            apps.get_app_config(app_label).get_model(model_name)
            for app_label, app_data in SEARCH_FIELDS.items()
            for model_name in app_data
        ]
        from .search_utils import update_spellfix
        for i in models:
            self.index_model(i)
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
            whits = set((i for i in whits.iterator()))
            print(f'with hits: {len(whits)} / total: ', end='', flush=True)
        else:
            whits = set()

        qs = model.objects.all()
        print(f'{qs.count()} [OK]')

        def searchable_objs():
            fs = SEARCH_FIELDS[model._meta.app_label][model._meta.model_name]
            for obj in qs:
                for field_name in fs:
                    txt = getattr(obj, field_name)
                    if txt is None or txt == '':
                        continue

                    yield self.model(
                        text=txt,
                        has_hit=(obj.pk in whits),
                        field=field_name,
                        content_object=obj,
                    )

        self.bulk_create(searchable_objs(), batch_size=10000)


class UniqueWordManager(Loader):
    @atomic_dry
    def reindex(self):
        """
        Populate table with unique, lower-case words from Searchable.text
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


class DatasetManager(Manager):
    def get_queryset(self):
        return super().get_queryset().filter(private=False)


class SampleManager(OmicsSampleManager):
    def get_queryset(self):
        return super().get_queryset().filter(dataset__private=False)
