from django.conf import settings
from django.db.transaction import atomic

from mibios.umrad.manager import QuerySet
from .utils import Timestamper


class SampleQuerySet(QuerySet):
    def check_metagenomic_data(self):
        """
        Return samples for which there is metagenomic data missing

        Also checks for consistency of the flags.
        """
        flags = [i for i, _ in self._manager.get_metagenomic_loader_script()]
        flags = ['metag_pipeline_reg'] + flags

        qs = self.filter(sample_type='metagenome')

        blocklist = self._manager.get_blocklist().values_list('pk', flat=True)
        if blocklist:
            print(f'{qs.filter(pk__in=blocklist).count()} metagenomic samples '
                  f'in blocklist')
            qs = qs.exclude(pk__in=blocklist)

        total = qs.count()
        qs = qs.filter(metag_pipeline_reg=True)
        reg_count = qs.count()
        print(f'Of {total} metagenomic samples, {reg_count} are registered, of those:')  # noqa: E501

        # flag consistency: A False flag must not be followed by one set True
        # for counting, we're only interested in registered samples
        all_missing_count = 0
        some_missing_count = 0
        for i in qs:
            some_false = False
            some_true = False
            for j in flags:
                if getattr(i, j):
                    if j != 'metag_pipeline_reg':
                        some_true = True
                    if some_false:
                        msg = f'Inconsistency with {i.sample_id} (pk={i.pk}): '
                        for k in flags:
                            msg += f'\n   {repr(getattr(i, k)):<5} {k}'
                        msg += '\n'
                        raise RuntimeError(msg)
                    continue
                else:
                    some_false = True

            # count registered only:
            if i.metag_pipeline_reg and some_false:
                if some_true:
                    some_missing_count += 1
                else:
                    all_missing_count += 1

        print(f'   all missing: {all_missing_count}')
        print(f'  some missing: {some_missing_count}')
        complete_count = reg_count - all_missing_count - some_missing_count
        print(f'      complete: {complete_count}')

        # all ok, return queryset with missing data
        return qs.filter(**{flags[-1]: False}).order_by('pk')

    def load_metagenomic_data(self, skip_check=False):
        """
        Load all metagenomics data
        """
        if skip_check:
            samples = self
        else:
            samples = self.check_metagenomic_data()

        # get number of stages
        total_stages = 0
        script = self._manager.get_metagenomic_loader_script()
        for _, funcs in script:
            if callable(funcs):
                total_stages += 1
            else:
                total_stages += len(funcs)

        template = f'[ {{sample}} {{{{stage}}}}/{total_stages} {{{{{{{{timestamp}}}}}}}} ]  '  # noqa: E501
        for num, sample in enumerate(samples):
            print(f'{len(samples) - num} samples to go...')
            stage = 1
            for flag, funcs in script:
                if callable(funcs):
                    funcs = [funcs]

                if getattr(sample, flag):
                    stage += len(funcs)
                    continue

                t = template.format(sample=sample.sample_id)

                # if not Alignment.loader.get_file(i).is_file():
                #    print(f'No m8 file: {i} skipping...')
                #    continue
                with atomic():
                    for fn in funcs:
                        timestamper = Timestamper(
                            template=t.format(stage=stage),
                            file_copy=settings.METAGENOMIC_LOADING_LOG,
                        )
                        with timestamper:
                            fn(sample)
                        stage += 1

                    if not getattr(sample, flag):
                        # not all functions will set the progress flag
                        setattr(sample, flag, True)
                        sample.save()

            print(f'Sample {sample} done!', end='  ')
        print()
