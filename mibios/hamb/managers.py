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
    SAMPLE_ID_PREFIX = 'sa'

    def get_last_sample_id_num(self):
        nums = (
            int(i.removeprefix(self.SAMPLE_ID_PREFIX))
            for i in self.values_list('sample_id', flat=True)
        )
        return max(nums, default=0)

    @atomic_dry
    def load_bio173_data(self, file):
        Dataset = self.model._meta.get_field('dataset').related_model
        Host = self.model._meta.get_field('host').related_model
        hosts = Host.objects.in_bulk(field_name='label')
        objs = []

        dataset = Dataset.objects.get(label='Bio173')
        id_num = self.get_last_sample_id_num()

        with open(file) as ifile:
            print(f'Reading {ifile.name} ...', end='', flush=True)
            head = ifile.readline().rstrip('\n').split('\t')
            for lnum, line in enumerate(ifile, start=2):
                row = line.rstrip('\n').split('\t')
                row = dict(zip(head, row, strict=True))
                id_num += 1

                objs.append(self.model(
                    dataset=dataset,
                    sample_id=f'{self.SAMPLE_ID_PREFIX}{id_num}',
                    label=row['Fecalsample'] or row['Name'],
                    host=hosts.get(row['Participant'], None),
                    source_material='' if row['Control'] else 'feces',
                    sample_type='amplicon',
                    amplicon_target='16S V4',
                    control=row['Control'],
                ))
        print(f'{len(objs)} [OK]')

        print('Validating objects... ', end='', flush=True)
        for i in objs:
            i.full_clean()
        print('[OK]')

        self.bulk_create(objs)

    @atomic_dry
    def load_predict1(self, shared_file):
        """
        Load all predict1 data
        """
        dataset_name = 'predict1'

        ASV = apps.get_model('omics', 'ASV')
        ASVAbundance = apps.get_model('omics', 'ASVAbundance')
        Dataset = self.model._meta.get_field('dataset').related_model

        dset_qs = Dataset.objects.filter(label=dataset_name)

        if dset_qs.exists():
            print('Deleting existing data... ', end='', flush=True)
            counts = dset_qs.delete()
            print(f'{counts} [OK]')

        print('Creating dataset record... ', end='', flush=True)
        dset = Dataset.objects.create(label=dataset_name, dataset_id=2)
        print('[OK]')

        print('Retrieving ASVs... ', end='', flush=True)
        all_asvs = ASV.objects.in_bulk(field_name='accession')
        print(f'{len(all_asvs)} [OK]')

        data = {}  # map sample name to count list
        print('Reading shared file... ', end='', flush=True)
        with open(shared_file) as ifile:
            _, _, _, *asvs = ifile.readline().split()
            for i in asvs:
                if i not in all_asvs:
                    raise RuntimeError(f'ASV not found in DB: {i}')
            for lnum, line in enumerate(ifile, start=2):
                _, sample, _, *counts = line.split()
                if len(counts) != len(asvs):
                    raise RuntimeError(f'bad column count at line {lnum}')
                try:
                    data[sample] = [int(i) for i in counts]
                except ValueError as e:
                    raise RuntimeError(
                        f'failed parsing count at line {lnum}: {e}'
                    ) from e

        if len(data) != (lnum - 1):
            raise RuntimeError('sample name duplicate?')
        print(f'{len(data)} [OK]')

        first_id_num = self.get_last_sample_id_num() + 1
        samples = {
            i: self.model(
                label=i,
                dataset=dset,
                sample_id=f'{self.SAMPLE_ID_PREFIX}{idnum}',
                sample_type='amplicon',
                amplicon_target='16S V4',
            )
            for idnum, i
            in enumerate(data, start=first_id_num)
        }
        for i in samples.values():
            i.full_clean()

        print(f'Allocating sample id numbers: {first_id_num} to '
              f'{first_id_num + len(samples)}')
        self.bulk_create(samples.values())

        print('Compiling abundance... ', end='', flush=True)
        abund_objs = []
        for sample_name, counts in data.items():
            total = sum(counts)
            for asv_accn, count in zip(asvs, counts, strict=True):
                if count == 0:
                    continue
                abund_objs.append(ASVAbundance(
                    sample=samples[sample_name],
                    asv=all_asvs[asv_accn],
                    count=count,
                    relabund=count / total,
                ))
        print(f'{len(abund_objs)} [OK]')

        print('Saving abundance... ', end='', flush=True)
        ASVAbundance.objects.bulk_create(abund_objs)
        print('[OK]')


_incl_tax_asv_map = None


def get_taxon_asv(taxnode):
    global _incl_tax_asv_map
    if _incl_tax_asv_map is None:
        ASV = apps.get_model('omics', 'ASV')
        _incl_tax_asv_map = ASV.objects.get_tax_mapping()
    return _incl_tax_asv_map.get(taxnode.pk, [])
