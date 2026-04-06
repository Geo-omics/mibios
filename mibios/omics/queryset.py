from collections import Counter
from itertools import chain, groupby, product
import os
from pathlib import Path
import traceback
from urllib.parse import quote_plus

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.transaction import atomic, set_rollback
from django.utils.module_loading import import_string

from mibios import __version__ as version
from mibios.umrad.manager import QuerySet
from mibios.umrad.utils import atomic_dry, ProgressPrinter

from .managers import fkmap_cache_reset
from .utils import gentle_int, NoJobParameters, Timestamper


class AccessMixin:
    """ mixin for queryset for models with access field """
    def exclude_private(self, credentials=None):
        """
        Exclude samples of private datasets unless user is member of allowed
        group.

        credentials:
            This can be auth.User instance, AnonymousUser, or list of Group PKs
            or None.
        """
        if connection.vendor == 'postgresql':
            return self.filter(access__overlap=self._get_groupids(credentials))
        else:
            # fallback for sqlite (noop)
            return self

    @classmethod
    def _get_groupids(cls, credentials):
        """
        Helper to turn a User into sorted group ids
        """
        try:
            group_manager = credentials.groups
        except AttributeError:
            if credentials is None:
                groupids = set()
            else:
                # assume credentials is list of Group PKs
                groupids = set(credentials)
        else:
            groupids = set(group_manager.values_list('pk', flat=True))

        # ensure we got a 0 and it's sorted
        groupids.add(0)
        return tuple(sorted(groupids))


class DataTrackingQuerySet(QuerySet):
    def undo(self, fake=False, dry_run=False):
        """
        Undo each associated job and erase tracking info and file records

        fake [bool]:
            Undo the job accounting (remove tracking tickets and files) but
            don't actually run the jobs' undo methods, so any omics data will
            remain in the DB.  Do this to reverse stale tracking info.
            Remaining omics data may have to be deleted manually.  This is has
            nothing to do with the dry_run option.

        dry_run [bool]:
            Run everything but roll back any changes to the DB.  Any side
            effect that the the jobs' undo methods have may still occurr.  This
            has nothing to do with the fake option, both can be set
            independently.
        """
        # dry run wrapper:  On a dry run the outermost transaction is to be
        # rolled back at the very end so that the individual undos can see each
        # other effects.  On a real run we want to successful individual undos
        # remain committed even if a later undo crashes.
        if dry_run:
            with atomic():
                retv = self._undo(fake=fake)
                set_rollback(True)
            return retv
        else:
            return self._undo(fake=fake)

    def _undo(self, fake=False):
        """ Wrapped by undo() to do the actual work """
        # sort objects in reverse job dependency order
        jobs = import_string('mibios.omics.tracking.registry').jobs.values()
        rank = {job: rank for rank, job in enumerate(reversed(jobs))}
        pool = sorted(self, key=lambda x: rank[type(x.job)])
        totals = Counter()

        for i in pool:
            delcounts = i.undo(fake=fake)
            totals.update(delcounts)

        return totals


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
        SeqSample = apps.get_model('omics', 'SeqSample')
        samples = SeqSample.objects.exclude_private(credentials)
        return self.filter(sample__in=samples)


class LoadMixin:
    """
    Mixin for QuerySet to orchestrate loading data

    This requires a corresponding models.DataTracking table for our model.
    """

    def get_ready(self, only=None, sort_by_subject=False, verbose=False):
        """
        Return Job instances which are ready to go (but not yet done).

        only list:
            Only return jobs of the given type.  A list of Job classes or class
            names may be passed.  If None, the default, then all our samples'
            ready jobs are returned.
        sort_by_subject:
            Sort jobs by sample.  The default is to sort jobs by the flag, in
            order as defined in SampleTracking.Flag.
        verbose [bool]:
            If True, then print non-ready jobs with status indicated and list
            of missing files.
        """
        DataTracking = self.model._meta.get_field('tracking').related_model
        job_registry = import_string('mibios.omics.tracking.registry')
        Status = import_string('mibios.omics.tracking.Status')

        job_registry.clear_cache()

        if only is not None:
            _only = set()
            for i in only:
                if isinstance(i, str):
                    try:
                        _only.add(job_registry.jobs[i])
                    except KeyError as e:
                        raise ValueError('not a job class name: {i}') from e
                else:
                    if i in job_registry.jobs.values():
                        _only.add(i)
                    else:
                        raise ValueError('not a job class: {i}')
            only = _only

        qs = self
        blocklist = \
            self._manager.get_omics_blocklist().values_list('pk', flat=True)
        if blocklist:
            print(f'{qs.filter(pk__in=blocklist).count()} samples '
                  f'in blocklist')
            qs = qs.exclude(pk__in=blocklist)

        qs = qs.filter(tracking__flag=DataTracking.Flag.PIPELINE)
        qs = qs.prefetch_related('tracking')

        if sort_by_subject:
            jobs = []
        else:
            jobs = {i: [] for i in DataTracking.Flag}

        print(f'Checking {self.model._meta.verbose_name_plural} for ready data...')
        no_param_fails = []
        for subject in ProgressPrinter(length=qs.count())(qs):
            ready_jobs = []
            for tr in subject.tracking.all():
                try:
                    tr_job = tr.job
                except NoJobParameters:
                    # is a parametric job, but instantiation failed e.g.
                    # because data is gone
                    no_param_fails.append(tr)
                    continue

                for job in tr_job.before:
                    if only is not None and type(job) not in only:
                        continue
                    if job in ready_jobs:
                        continue
                    if job.is_ready():
                        ready_jobs.append(job)
                    elif verbose:
                        print(f'not ready: {job}')
                        if Status.MISSING in job.status():
                            for missing_file in job.status()[Status.MISSING]:
                                print(f'   missing: {missing_file}')

            if sort_by_subject:
                jobs += ready_jobs
            else:
                for i in ready_jobs:
                    jobs[i.flag].append(i)

        if no_param_fails:
            fail_list = ', '.join(str(i) for i in no_param_fails[:3])
            if len(no_param_fails) > 3:
                fail_list += ', ...'
            print(
                f'[NOTICE] Job parameterization failed for {len(no_param_fails)} '
                f'existing tracking tickets:\n --> {fail_list}'
            )

        if sort_by_subject:
            return jobs
        else:
            return list(chain(*jobs.values()))

    @gentle_int
    def load_omics_data(self, jobs=None, follow_up_jobs=True,
                        publish_files=True, dry_run=False):
        """
        Run ready omics data loading jobs for each object.

        jobs list:
            list of jobs to process.  If this is None, then all possible jobs
            with status READY will be processed.  Jobs not belonging to the
            queryset will be skipped.

        follow_up_jobs bool:
            If this is True, the default, then upon completion of a job, other
            jobs that were not ready before but are ready now, will also be
            run.

        publish_files bool:
            If True, the default, then, after a run, the loaded files will also
            be published (saved to local and/or globus storage.)

        This is a wrapper to handle the dry_run parameter.  In a dry run
        everything is done inside an outer transaction that is then rolled
        back, as usual.  But in a production run we don't want the outer
        transaction as to not lose work of successful jobs when we later crash.
        """
        kwargs = dict(
            jobs=jobs,
            follow_up_jobs=follow_up_jobs,
            publish_files=publish_files,
        )
        if dry_run:
            with atomic():
                ret = self._load_omics_data(**kwargs)
                set_rollback(True)
                return ret
        else:
            return self._load_omics_data(**kwargs)

    def _load_omics_data(self, jobs=None, follow_up_jobs=True,
                         publish_files=True):
        timestamper = Timestamper(
            template='[ {timestamp} ]  ',
            file_copy=settings.OMICS_LOADING_LOG,
        )

        model_name_pl = self.model._meta.verbose_name_plural

        with timestamper:
            print(f'Loading omics data / version: {version}')
            if jobs is None:
                jobs_todo = self.get_ready(sort_by_subject=True)
                if not jobs_todo:
                    print(f'NOTICE: no ready jobs for these {model_name_pl}')
                    return
            else:
                subject_set = set(self)
                jobs_todo = [i for i in jobs if i.subject in subject_set]
                if len(jobs) > len(jobs_todo):
                    print(f'WARNING: {len(jobs) - len(jobs_todo)} given jobs'
                          f'are for other {model_name_pl} and will be skipped')
                if not jobs_todo:
                    print('NOTICE: there are no jobs to be done')
                    return

            # some of the accounting below makes more sense if jobs are sorted
            # by subject, but the logic should still work even if later jobs
            # return to same subject.
            jobs_per_subject = [
                (subject, list(job_grp))
                for subject, job_grp
                in groupby(jobs_todo, key=lambda x: x.subject)
            ]
            subject_count = len(set((i for i, _ in jobs_per_subject)))
            if len(self):
                if no_job_subject_count := len(self) - subject_count:
                    if jobs:
                        print(f'{no_job_subject_count} {model_name_pl} have '
                              f'no jobs')
                    else:
                        print(f'{no_job_subject_count} {model_name_pl} have no'
                              f' ready jobs')
            print(f'Will run a total of {len(jobs_todo)} jobs across '
                  f'{subject_count} {model_name_pl}.')

        File = apps.get_model('omics', 'File')
        template = '[ {subject} {{stage}}/{{total_stages}} {{{{timestamp}}}} ]  '  # noqa: E501
        fkmap_cache_reset()
        for num, (subject, job_grp) in enumerate(jobs_per_subject):
            print(f'{len(jobs_todo) - num} {model_name_pl} to go...')
            abort_subject = False
            # for flag, funcs in script:  # OLD
            stage = 0
            while job_grp:
                job = job_grp.pop(0)
                stage += 1

                t = template.format(subject=subject.accession)

                timestamper = Timestamper(
                    template=t.format(stage=stage, total_stages=stage + len(job_grp)),  # noqa: E501
                    file_copy=settings.OMICS_LOADING_LOG,
                )
                with atomic(), timestamper:
                    if stage == 1:
                        print(f'--> {type(job).__name__} | {job.subject.accession}')

                    try:
                        job()
                    except KeyboardInterrupt as e:
                        print(repr(e))
                        raise
                    except Exception as e:
                        msg = []
                        if isinstance(e, ValidationError):
                            # pretty-print expected errors
                            msg.append('[ValidationError]')
                            for key, msglist in e:
                                for m in msglist:
                                    msg.append(f' -> {key}: {m}')
                        else:
                            msg.append(
                                f'{type(e).__name__} "{e}": on '
                                f'{subject.accession} at or near {job.run=}'
                            )
                        print(*msg, sep='\n')
                        # If we're configured to write a log file, then print
                        # the stack to a special FAIL.log file and continue
                        # with the next subject. This optimizes for the case
                        # that the error is caused by occasional unusual data
                        # for individual subjects and not a regular bug, which
                        # would trigger on every subject/sample.
                        log = settings.OMICS_LOADING_LOG
                        if not log:
                            raise
                        faillog = Path(log).with_suffix(
                            f'.{subject.accession}.FAIL.log'
                        )
                        with faillog.open('w') as ofile:
                            ofile.write('\n'.join(msg) + '\n')
                            traceback.print_exc(file=ofile)
                        print(f'(see traceback at {faillog})')
                        # skip to next subject, do not set the flag, job() is
                        # assumed to have rolled back any its own changes to
                        # the DB
                        abort_subject = True
                        break
                    else:
                        if publish_files:
                            file_pks = (i.pk for i in job.files)
                            files = File.objects.filter(pk__in=file_pks)
                            try:
                                files.update_storage()
                            except OSError as e:
                                # not fatal, file publishing can also be done
                                # later, manually
                                print(f'[ERROR] was trying to publish files: '
                                      f'{e}')

                        if follow_up_jobs:
                            # add newly ready jobs
                            for i in reversed(job.before):
                                if i not in job_grp:
                                    if i.is_ready(use_cache=False):
                                        job_grp.insert(0, i)

            if abort_subject:
                print(f'Aborting {subject.accession}!', end=' ')
            else:
                print(f'{self.model._meta.verbose_name} {subject.accession} '
                      f'done!', end=' ')
        print()


class BaseDatasetQuerySet(AccessMixin, LoadMixin, QuerySet):
    @atomic_dry
    def start_tracking(self):
        """
        Check if dataset has a presence at the omics pipeline

        Update tracking info unless dry_run is True.
        """
        DatasetTracking = self.model._meta.get_field('tracking').related_model

        stats = {'total': self.count()}

        qs = self.filter(
            sample__seqsample__tracking__flag=DatasetTracking.Flag.PIPELINE
        )
        qs = qs.distinct()
        stats['has_samples'] = qs.count()
        stats['new'] = 0

        for obj in qs:
            if obj.project_dir.is_dir():
                tr, new = DatasetTracking.objects.get_or_create(
                    subject=obj,
                    flag=DatasetTracking.Flag.PIPELINE,
                )
                if new:
                    stats['new'] += 1
                else:
                    tr.save()  # update timestamp
            else:
                print(f'[WARNING] no such directory: {obj.project_dir}')

        return stats


class SeqSampleQuerySet(AccessMixin, LoadMixin, QuerySet):

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
                path = Path(file.file_pipeline.path)
                if not path.exists():
                    continue

                file_count += 1
                found_a_file = True
                with path.open() as ifile:
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

    def update_access(self):
        """
        Make values of access field consistent with parent samples and Datasets

        Delegates to the parents' queryset method update_access()
        """
        Sample = self.model._meta.get_field('parent').related_model
        Sample.objects.filter(seqsample__in=self).update_access()
