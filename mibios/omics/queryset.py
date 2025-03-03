from itertools import chain, groupby, product
import os
from pathlib import Path
import traceback
from urllib.parse import quote_plus

from django.apps import apps
from django.conf import settings
from django.db.transaction import atomic, set_rollback
from django.utils.module_loading import import_string

from mibios import __version__ as version
from mibios.umrad.manager import QuerySet

from . import get_sample_model
from .managers import fkmap_cache_reset
from .utils import gentle_int, Timestamper


class FileQuerySet(QuerySet):
    def extra_context(self):
        """
        Get additional view/template context to add Globus directory link for
        file listings

        To be called by View.get_context_data() as e.g.

            if hasattr(self.object_list, 'extra_context'):
                ctx.update(self.object_list.extra_context())
        """
        ctx = {}
        if globus_base := settings.GLOBUS_FILE_APP_URL_BASE:
            # the files' directories, split into parts:
            # FIXME: this does an additional DB query, unecessary but no idea
            # how to avoid (still an issue?)
            paths = [
                Path(i.file_globus.name).parts[:-1]
                for i in self if i.file_globus
            ]
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

    def update_storage(self, local=True, globus=True, dry_run=False):
        """
        Set file fields and ensure files are in storage.

        dry_run: If True, only report what would be done.
        """
        field_names = []
        if local:
            field_names.append('file_local')
        if globus:
            field_names.append('file_globus')
        if not field_names:
            return

        changed_objs = set()
        for obj, field_name in product(self, field_names):
            changed = obj.update_storage(field_name, dry_run=dry_run)
            if changed:
                changed_objs.add(obj)

        print(f'Updating {len(changed_objs)} omics.File objects ...', end='',
              flush=True)
        if not dry_run:
            self.model.objects.bulk_update(changed_objs, field_names)
        if dry_run:
            print('[dryrun]')
        else:
            print('[OK]')

    def exclude_private(self, credentials=None):
        Sample = get_sample_model()
        samples = Sample.objects.exclude_private(credentials)
        return self.filter(sample__in=samples)


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
        blocklist = \
            self._manager.get_omics_blocklist().values_list('pk', flat=True)
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
    def load_omics_data(self, jobs=None, follow_up_jobs=True, dry_run=False):
        """
        Run ready omics data loading jobs for these samples.

        jobs list:
            list of jobs to process.  If this is None, then all possible jobs
            with status READY will be processed.  Jobs not beloging to the
            queryset will be skipped.

        follow_up_jobs bool:
            If this is True, the default, then upon completion of a job, other
            jobs that were not ready before but are ready now, will also be
            run.

        This is a wrapper to handle the dry_run parameter.  In a dry run
        everything is done inside an outer transaction that is then rolled
        back, as usual.  But in a production run we don't want the outer
        transaction as to not lose work of successful jobs when we later crash.
        """
        if dry_run:
            with atomic():
                ret = self._load_omics_data(jobs=jobs)
                set_rollback(True)
                return ret
        else:
            return self._load_omics_data(jobs=jobs)

    def _load_omics_data(self, jobs=None, follow_up_jobs=True):
        timestamper = Timestamper(
            template='[ {timestamp} ]  ',
            file_copy=settings.OMICS_LOADING_LOG,
        )

        with timestamper:
            print(f'Loading omics data / version: {version}')
            if jobs is None:
                jobs_todo = self.get_ready(sort_by_sample=True)
                if not jobs_todo:
                    print('NOTICE: no ready jobs for these samples')
                    return
            else:
                sample_set = set(self)
                jobs_todo = [i for i in jobs if i.sample in sample_set]
                if len(jobs) > len(jobs_todo):
                    print(f'WARNING: {len(jobs) - len(jobs_todo)} given jobs'
                          f'are for other samples and will be skipped')
                if not jobs_todo:
                    print('NOTICE: there are no jobs to be done')
                    return

            # some of the accounting below makes more sense if jobs are sorted
            # by sample, but the logic should still work even if later jobs
            # return to same sample.
            jobs_per_sample = [
                (sample, list(job_grp))
                for sample, job_grp
                in groupby(jobs_todo, key=lambda x: x.sample)
            ]
            sample_count = len(set((i for i, _ in jobs_per_sample)))
            if len(self):
                if no_job_sample_count := len(self) - sample_count:
                    if jobs:
                        print(f'{no_job_sample_count} samples have no jobs')
                    else:
                        print(f'{no_job_sample_count} samples have no ready '
                              f'jobs')
            print(f'Will run a total of {len(jobs_todo)} jobs across '
                  f'{sample_count} samples.')

        File = apps.get_model('omics', 'File')
        template = '[ {sample} {{stage}}/{{total_stages}} {{{{timestamp}}}} ]  '  # noqa: E501
        fkmap_cache_reset()
        for num, (sample, job_grp) in enumerate(jobs_per_sample):
            print(f'{len(jobs_todo) - num} samples to go...')
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
                    if stage == 1:
                        print(f'--> {type(job).__name__}')

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
                        # publish files
                        file_pks = (i.pk for i in job.files)
                        files = File.objects.filter(pk__in=file_pks)
                        try:
                            files.update_storage()
                        except OSError as e:
                            # file publishing can be done manually later
                            print(f'[ERROR] trying to publish files: {e}')

                        if follow_up_jobs:
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

    def exclude_private(self, credentials=None):
        # this is a no-op in omics, as FileQueryset depends on it.  Inheriting
        # classes implement it as needed.
        return self
