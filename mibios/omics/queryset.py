import os
import traceback

from django.conf import settings
from django.db.transaction import atomic

from mibios import __version__ as version
from mibios.umrad.manager import QuerySet
from .managers import fkmap_cache_reset
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
        timestamper = Timestamper(
            template='[ {timestamp} ]  ',
            file_copy=settings.METAGENOMIC_LOADING_LOG,
        )
        with timestamper:
            print(f'Loading metagenomic data / version: {version}')
            if skip_check:
                samples = self
                print(f'{len(self)} samples')
            else:
                samples = self.check_metagenomic_data()

            # get number of stages
            total_stages = 0
            script = self._manager.get_metagenomic_loader_script()
            print('Stages / flags: ')
            for flag, funcs in script:
                if callable(funcs):
                    total_stages += 1
                    print(f'  {total_stages} / {flag} ')
                else:
                    total_stages += len(funcs)
                    print(f'  {total_stages - len(funcs) + 1}-{total_stages} / {flag} ')  # noqa: E501

        template = f'[ {{sample}} {{{{stage}}}}/{total_stages} {{{{{{{{timestamp}}}}}}}} ]  '  # noqa: E501
        fkmap_cache_reset()
        for num, sample in enumerate(samples):
            print(f'{len(samples) - num} samples to go...')
            stage = 1
            abort_sample = False
            for flag, funcs in script:
                if callable(funcs):
                    funcs = [funcs]

                if getattr(sample, flag):
                    stage += len(funcs)
                    continue

                t = template.format(sample=sample.sample_id)

                with atomic():
                    for fn in funcs:
                        timestamper = Timestamper(
                            template=t.format(stage=stage),
                            file_copy=settings.METAGENOMIC_LOADING_LOG,
                        )
                        with timestamper:
                            try:
                                fn(sample)
                            except Exception as e:
                                # If we're configured to write a log file, then
                                # print the stack to a special FAIL.log file
                                # and continue with the next sample. This
                                # assumes the error is caused not by a regular
                                # bug but by occasional unusual data for
                                # individual samples.
                                path = settings.METAGENOMIC_LOADING_LOG
                                if not path:
                                    raise
                                path, _, _ = path.rpartition('.')
                                path = f'{path}.{sample.sample_id}.FAIL.log'
                                msg = (f'FAIL: {e.__class__.__name__} {e} on '
                                       f'{sample.sample_id} at {fn=}')
                                print(msg)
                                with open(path, 'w') as ofile:
                                    ofile.write(msg + '\n')
                                    traceback.print_exc(file=ofile)
                                print(f'see traceback at {ofile.name}')
                                # skip to next sample, do not set the flag,
                                # fn() is assumed to have rolled back any
                                # its own changes to the DB
                                abort_sample = True
                                break
                        stage += 1

                    if abort_sample:
                        break

                    if not getattr(sample, flag):
                        # not all functions will set the progress flag
                        setattr(sample, flag, True)
                        sample.save()

            if abort_sample:
                print(f'Aborting {sample.sample_id}/{sample}!', end=' ')
            else:
                print(f'Sample {sample.sample_id}/{sample} done!', end=' ')
        print()

    def ur100_accession_crawler(self, outname=None, verbose=False):
        """ Extract UniRef100 accessions from omics data """
        if not outname:
            outname = 'ur100.accessions.txt'

        FILE_PATS = [
            # filename pattern, 1-based ur100 column number
            ('{sample_id}_tophit_report', 1),
            ('{sample_id}_contig_tophit_report', 1),
        ]

        PREFIXES = ['UNIREF100_', 'UniRef100_']

        accns = set()
        old_total = 0
        file_count = 0
        for sample in self:
            base = sample.get_metagenome_path()
            found_a_file = False
            for pat, col in FILE_PATS:
                path = base / pat.format(sample_id=sample.sample_id)
                if not path.exists():
                    continue

                file_count += 1
                found_a_file = True
                with open(path) as ifile:
                    os.posix_fadvise(
                        ifile.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL
                    )
                    os.posix_fadvise(
                        ifile.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                    )
                    for linenum, line in enumerate(ifile, start=1):
                        row = line.rstrip('\n').split('\t', maxsplit=col)
                        value = row[col - 1]

                        for i in PREFIXES:
                            if value.startswith(i):
                                value = value.removeprefix(i)
                        accns.add(value)
                if verbose:
                    cur_total = len(accns)
                    print(
                        f'{cur_total:>10} {sample.sample_id:>9}:'
                        f'{path.name:<33}\t'
                        f'{linenum:>7} new:{cur_total - old_total:>8}'
                    )
                    old_total = cur_total
            if verbose and not found_a_file:
                print(f'{sample.sample_id}: no files found')

        print(f'Searched {file_count} files, found {len(accns)} distinct '
              f'accessions.')

        with open(outname, 'w') as ofile:
            for i in sorted(accns):
                ofile.write(f'{i}\n')
        print(f'UniRef100 accessions written to {outname}')
