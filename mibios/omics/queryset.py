from itertools import chain, groupby
import os
from pathlib import Path
import traceback
from urllib.parse import quote_plus

from django.conf import settings
from django.db.transaction import atomic
from django.utils.module_loading import import_string

from mibios import __version__ as version
from mibios.umrad.manager import QuerySet
from .managers import fkmap_cache_reset
from .utils import gentle_int, Timestamper


class FileQuerySet(QuerySet):
    def extra_context(self):
        """
        Get additional view/template context

        Call in View.get_context_data() as e.g.
        if hasattr(self.object_list, 'extra_context'):
            ctx.update(self.object_list.extra_context())
        """
        ctx = {}
        globus_base = settings.GLOBUS_FILE_APP_URL_BASE
        if globus_base:
            # the files' directories, split into parts:
            # FIXME: this does an additional DB query, unecessary but no idea
            # how to avoid (still an issue?)
            paths = [i.relpublic.parts[:-1] for i in self if i]
            if paths:
                common_parts = []
                for parts in zip(*paths):
                    if len(set(parts)) == 1 and len(parts) == len(paths):
                        # part is present in all paths
                        common_parts.append(parts[0])
                    else:
                        break

                common_path = Path(*common_parts)

                globus_url = globus_base + quote_plus(str(common_path))
                item = {
                    'label': 'Globus File Manager',
                    'url': globus_url,
                }
                ctx['extra_navigation'] = [item]
        return ctx


class SampleQuerySet(QuerySet):
    def get_ready(self, sort_by_sample=False):
        """
        Return Job instances which are ready to go (but not yet done).

        Params:
        sort_by_sample:
            Sort jobs by sample.  The default is to sort jobs by the flag, in
            order as defined in SampleTracking.Flag.
        """
        SampleTracking = import_string('mibios.omics.models.SampleTracking')

        qs = self
        blocklist = self._manager.get_blocklist().values_list('pk', flat=True)
        if blocklist:
            print(f'{qs.filter(pk__in=blocklist).count()} samples '
                  f'in blocklist')
            qs = qs.exclude(pk__in=blocklist)

        qs = qs.filter(tracking__flag=SampleTracking.Flag.PIPELINE)
        qs = qs.prefetch_related('tracking')

        if sort_by_sample:
            jobs = []
        else:
            jobs = {i: [] for i in SampleTracking.Flag}

        for sample in qs:
            ready_jobs = []
            for tr in sample.tracking.all():
                for job in tr.job.before:
                    if job.is_ready() and job not in ready_jobs:
                        ready_jobs.append(job)
            if sort_by_sample:
                jobs += ready_jobs
            else:
                for i in ready_jobs:
                    jobs[i.flag].append(i)

        if sort_by_sample:
            return jobs
        else:
            return list(chain(*jobs.values()))

    @gentle_int
    def load_omics_data(self):
        """
        Load all metagenomics data
        """
        timestamper = Timestamper(
            template='[ {timestamp} ]  ',
            file_copy=settings.OMICS_LOADING_LOG,
        )
        with timestamper:
            print(f'Loading omics data / version: {version}')
            jobs = self.get_ready(sort_by_sample=True)
            jobs_total = len(jobs)
            jobs = {
                sample: list(job_grp)
                for sample, job_grp
                in groupby(jobs, key=lambda x: x.sample)
            }
            print(f'samples: {len(jobs)} -- total jobs: {jobs_total}')

        template = '[ {sample} {{stage}}/{{total_stages}} {{{{timestamp}}}} ]  '  # noqa: E501
        fkmap_cache_reset()
        for num, (sample, job_grp) in enumerate(jobs.items()):
            print(f'{len(jobs) - num} samples to go...')
            abort_sample = False
            # for flag, funcs in script:  # OLD
            stage = 0
            while job_grp:
                job = job_grp.pop(0)
                stage += 1

                t = template.format(sample=sample.sample_id)

                timestamper = Timestamper(
                    template=t.format(stage=stage, total_stages=stage + len(job_grp)),  # noqa: E501
                    file_copy=settings.OMICS_LOADING_LOG,
                )
                with atomic(), timestamper:
                    try:
                        job()
                    except KeyboardInterrupt as e:
                        print(repr(e))
                        raise
                    except Exception as e:
                        msg = (f'FAIL: {e.__class__.__name__} "{e}": on '
                               f'{sample.sample_id} at or near {job.run=}')
                        print(msg)
                        # If we're configured to write a log file, then print
                        # the stack to a special FAIL.log file and continue
                        # with the next sample. This optimizes for the case
                        # that the error is caused by occasional unusual data
                        # for individual samples and not a regular bug, which
                        # would trigger on every sample.
                        log = settings.OMICS_LOADING_LOG
                        if not log:
                            raise
                        faillog = Path(log).with_suffix(
                            f'.{sample.sample_id}.FAIL.log'
                        )
                        with faillog.open('w') as ofile:
                            ofile.write(msg + '\n')
                            traceback.print_exc(file=ofile)
                        print(f'see traceback at {faillog}')
                        # skip to next sample, do not set the flag, fn() is
                        # assumed to have rolled back any its own changes to
                        # the DB
                        abort_sample = True
                        break
                    else:
                        # add newly ready jobs
                        for i in reversed(job.before):
                            if i not in job_grp:
                                if i.is_ready(use_cache=False):
                                    job_grp.insert(0, i)

            if abort_sample:
                print(f'Aborting {sample.sample_id}/{sample}!', end=' ')
            else:
                print(f'Sample {sample.sample_id}/{sample} done!', end=' ')
        print()

    def ur100_accession_crawler(self, outname=None, verbose=False):
        """ Extract UniRef100 accessions from omics data """
        if not outname:
            outname = 'ur100.accessions.txt'

        File = import_string('mibios.omics.models.File')
        FILETYPES = [
            # filetype, 1-based ur100 column number
            (File.Type.FUNC_ABUND, 1),
            # ('{sample_id}_contig_tophit_report', 1),
        ]

        PREFIXES = ['UNIREF100_', 'UniRef100_']

        accns = set()
        old_total = 0
        file_count = 0
        for sample in self:
            found_a_file = False
            for ftype, col in FILETYPES:
                try:
                    file = sample.get_omics_file(ftype)
                except Exception:
                    # e.g. sample w/o analysis_dir
                    continue
                if not file.path.exists():
                    continue

                file_count += 1
                found_a_file = True
                with open(file) as ifile:
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
                        f'{file.path.name:<33}\t'
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
