from django.core.management.base import BaseCommand

from mibios.glamr.models import Sample


OUTPUT = 'seagull_sample_metadata.csv'
SEP = '\t'

BASE_FIELDS = [
    'sample_id',
    'sample_name',
    'collection_timestamp',
    'collection_ts_partial',
    'latitude',
    'longitude',
]

EXCLUDE_FIELDS = [
    # technical field for omics app
    'tracking_id',
    'metag_pipeline_reg',
    'analysis_dir',
    'read_count',
    'reads_mapped_contigs',
    'reads_mapped_genes',
    # geomicrobio internal fields
    'sortchem',
    'notes',
]

COLUMN_HEADER = [
    'internal_dataset_id',
    'internal_sample_id',
    'upstream_sample_name',
    'sample_collection_timestamp',
    'timestamp_precision',
    'latitude',
    'longitude',
    'field_name',
    'unit',
    'field_value',
]


class Command(BaseCommand):
    help = ('Export data for transfer to Seagull. '
            'Currently this makes a long-form table csv file with sample meta '
            'data.')

    def handle(self, *args, **options):
        row_fields = [
            (i.name, i.verbose_name, getattr(i, 'unit', ''))
            for i in Sample._meta.get_fields()
            if i.concrete
            and i.name not in BASE_FIELDS
            and i.name not in EXCLUDE_FIELDS
            and i.name != 'id'
            and not i.name.endswith('_loaded')
            and not i.name.endswith('_ok')
        ]
        skip_count = 0
        with open(OUTPUT, 'w') as ofile:
            ofile.write(f'{SEP.join(COLUMN_HEADER)}\n')
            for i in Sample.objects.all():
                base_row = [i.dataset.dataset_id]
                for j in BASE_FIELDS:
                    val = getattr(i, j)
                    if val is None:
                        # skip sample
                        skip_count += 1
                        break
                    base_row.append(val)
                else:
                    base_row = [str(k) for k in base_row]
                    for field_name, verbose_name, unit in row_fields:
                        row = list(base_row)
                        row.append(verbose_name)
                        row.append(unit)
                        val = getattr(i, field_name)
                        if val is None:
                            val = ''
                        row.append(str(val))
                        row = SEP.join(row)
                        ofile.write(f'{row}\n')
        if skip_count:
            print(f'WARNING: {skip_count} samples skipped due to missing base '
                  f'field values')
        print(f'Saved to {OUTPUT}')
