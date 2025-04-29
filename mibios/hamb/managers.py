from django.apps import apps

from mibios.umrad.manager import Manager
from mibios.umrad.utils import atomic_dry


class HostManager(Manager):
    @atomic_dry
    def load_bio173_data(self, file):
        objs = []
        with open(file) as ifile:
            print(f'Reading {file}... ', end='', flush=True)
            head = ifile.readline().rstrip('\n').split('\t')
            for lnum, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                row = dict(zip(head, row, strict=True))
                objs.append(self.model(
                    label=row['Name'],
                    common_name='human',
                    age_years=row['Age'],
                    health_state='healthy',
                ))
        print(f' {len(objs)} [OK]')

        print('Validating objects... ', end='', flush=True)
        for i in objs:
            i.full_clean()
        print('[OK]')

        self.bulk_create(objs)


class SampleManager(Manager):
    @atomic_dry
    def load_bio173_data(self, file):
        Dataset = self.model._meta.get_field('dataset').related_model
        Host = self.model._meta.get_field('host').related_model
        hosts = Host.objects.in_bulk(field_name='label')
        objs = []

        dataset = Dataset.objects.get(label='Bio173')
        id_num = 0

        with open(file) as ifile:
            print(f'Reading {ifile.name} ...', end='', flush=True)
            head = ifile.readline().rstrip('\n').split('\t')
            for lnum, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                row = dict(zip(head, row, strict=True))
                id_num += 1
                objs.append(self.model(
                    dataset=dataset,
                    sample_id=f'sa{id_num}',
                    label=row['Fecalsample'] or row['Name'],
                    host=hosts.get(row['Participant'], None),
                    source_material='feces',
                    sample_type='amplicon',
                    amplicon_target='16S V4',
                ))
        print(f'{len(objs)} [OK]')

        print('Validating objects... ', end='', flush=True)
        for i in objs:
            i.full_clean()
        print('[OK]')

        self.bulk_create(objs)


_incl_tax_asv_map = None


def get_taxon_asv(taxnode):
    global _incl_tax_asv_map
    if _incl_tax_asv_map is None:
        ASV = apps.get_model('omics', 'ASV')
        _incl_tax_asv_map = ASV.objects.get_tax_mapping()
    return _incl_tax_asv_map.get(taxnode.pk, [])
