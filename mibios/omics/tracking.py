from collections import Counter
from enum import Enum
from graphlib import TopologicalSorter
from importlib import import_module
import inspect
import sys

from django.core.exceptions import ValidationError
from django.db.transaction import atomic

from mibios.umrad.utils import atomic_dry
from . import get_dataset_model
from .models import File, SeqSample
from .utils import NoJobParameters


Status = Enum('Status', ['READY', 'DONE', 'WAITING', 'MISSING'])


class BaseJob:
    model = None
    enabled = True
    flag = None
    after = None
    before = None
    required_files = None
    run = None
    undo = None

    def __init__(self, subject, tracking=None):
        """
        don't use this constructor directly, use e.g. for_subject() instead.
        """
        if self.model is not None and not isinstance(subject, self.model):
            raise TypeError(f'subject must be instance of {self.model} not '
                            f'{type(subject)}')

        if not self.is_compatible(subject):
            raise ValueError(f'{type(self)}: job can\'t handle {subject}')

        if tracking is None:
            if hasattr(subject, '_prefetched_objects_cache'):
                for i in subject._prefetched_objects_cache.get('tracking', []):
                    if i.flag == self.flag.value:
                        tracking = i
                        break
        else:
            if self.flag.value != tracking.flag:
                raise ValueError('flag mismatch')

        self.subject = subject
        self.tracking = tracking
        self.before = []
        self.after = []
        self._status = None

        self.params = self.get_params()
        if not self.params:
            raise NoJobParameters()

        # Add file parameters
        file_types = self.required_files or []
        self.files = []
        for kwargs in self.params:
            files = [
                self.subject.get_omics_file(i, **kwargs)
                for i in file_types
            ]
            # For historical reasons, a single file uses the 'file' key, when a
            # job has multiple files, keys are derived from the filetype's
            # name.  The job's run function's signature has to match these.
            if len(files) == 1:
                kwargs['file'] = files[0]
            elif len(files) > 1:
                kwargs['file'] = None
                kwargs.update({i.filetype_name.lower(): i for i in files})
            self.files += files

    _jobs = None
    """ class-level object holder for for_subject(); initialized to {} in registry """

    @classmethod
    def clear_cache(cls):
        cls._jobs = {}

    def __str__(self):
        if self.is_done():
            status = Status.DONE.name
        elif self.is_ready():
            status = Status.READY.name
        elif Status.WAITING in self.status():
            status = Status.WAITING.name
        elif Status.MISSING in self.status():
            status = Status.MISSING.name
        else:
            status = self.status()

        return f'{self.subject.accession}/{self.__class__.__name__} [{status}]'

    @atomic
    def __call__(self):
        """
        Run the job

        This calls the callable stored under the run attribute with the subject
        as positional argument and tracks the success in the tracking table.

        As part of running the job, this also tracks the files used and add a
        tracking entry.

        This will not check certain pre-conditions, e.g. one should call e.g.
        is_ready() beforehand.
        """
        if self.run is None:
            raise RuntimeError('job can not be run')
        if not self.enabled:
            raise RuntimeError('job is disabled')

        self.pre_check_files()

        retvals = []
        for kwargs in self.params:
            retv = self.run(self.subject, **kwargs)
            retvals.append(retv)

        self.post_check_files()

        if self.tracking is None:
            self.tracking = self.subject.tracking.model(
                flag=self.flag,
                subject=self.subject,
            )
        self.tracking.full_clean()
        self.tracking.save()

        # clear cached tracking data, then update status
        try:
            del self.subject._prefetched_objects_cache['tracking']
        except (AttributeError, KeyError):
            pass
        self.status(use_cache=False)

        if len(retvals) == 1:
            return retvals[0]
        else:
            return retvals

    def get_params(self):
        """
        Generate parameters for the several jobs if needed.

        Returns a list of dicts.  Inheriting classes should overwrite this if
        needed.  The default implementation returns a single empty dict,
        meaning the job runs once without special parameters.  If multiple
        dictionaries are returned then the callable stored under self.run will
        be called with each set of parameters passed as keyword arguments.
        Further, the files passed to each call of self.run get parameterized.

        If an empty list is returned, the job is deemed not to be ready and
        will be disappeared, useful to implement some autodetection scheme to
        generate jobs based to available data..
        """
        return [{}]

    @atomic_dry
    def run_undo(self, force=False):
        """
        Undo the effect of run()

        force [bool]:
            Skip job status check.  This option may lead to inconsistencies:
            jobs may be DONE before before jobs they depend on are marked DONE.

        This will also delete the tracking instance from the DB.  To undo a
        job, all jobs depending on this one, must be undone first.
        """
        if self.undo is None:
            raise NotImplementedError(
                f'{self}: The Job.undo attribute must be set to a callable '
                f'that takes the subject as argument.'
            )

        if not force:
            # guard rails
            if not self.is_done(use_cache=False):
                raise RuntimeError('to undo, job must be done first')
            for i in self.before:
                if i.is_done(use_cache=False):
                    raise RuntimeError(
                        f'a job that depends on this is already done: {i}'
                    )

        delcounts = Counter()
        for kwargs in self.params:
            retv = self.undo(self.subject, **kwargs)
            if isinstance(retv, Counter):
                delcounts.update(retv)

        if self.tracking is not None and self.tracking.id is not None:
            _, counts = self.tracking.delete()
            delcounts.update(counts)
        self.tracking = None
        self._status = None

        file_pks = [i.pk for i in self.files]
        _, counts = File.objects.filter(pk__in=file_pks).delete()
        delcounts.update(counts)

        return delcounts

    @classmethod
    def for_subject(cls, subject, tracking=None):
        """
        Get the job instance for given subject (a.k.a. object).

        Use this in place of the regular constructor to ensure that jobs are
        singleton per subject.
        """
        if subject not in cls._jobs:
            obj = cls(subject, tracking=tracking)
            # store obj now to avoid infinite recursion trying to instantiate
            # before and after jobs.
            cls._jobs[subject] = obj
            # FIXME: job declarations should eventually have some
            # conditionals, e.g. do next job only for some sampletype
            obj.after = []
            for jobcls in cls.after:
                if jobcls.is_compatible(subject):
                    try:
                        obj.after.append(jobcls.for_subject(subject))
                    except NoJobParameters:
                        pass
            obj.before = []
            for jobcls in cls.before:
                if jobcls.is_compatible(subject):
                    try:
                        obj.before.append(jobcls.for_subject(subject))
                    except NoJobParameters:
                        pass
        return cls._jobs[subject]

    @classmethod
    def is_compatible(cls, subject):
        """ tell if job is compatible with subject """
        if cls.model is None:
            return True
        else:
            return isinstance(subject, cls.model)

    def pre_check_files(self):
        """
        Check files before running the job
        """
        errors = {}
        for i in self.files:
            try:
                i.check_stat_field('file_pipeline')
            except ValidationError as e:
                errors = e.update_error_dict(errors)
            try:
                i.verify_with_pipeline()
            except ValidationError as e:
                errors = e.update_error_dict(errors)
        if errors:
            raise ValidationError(errors)

    def post_check_files(self):
        """
        Check files after running job

        1. Checks that files didn't changes while processing
        2. saves file records to database
        """
        errors = {}
        for i in self.files:
            try:
                # TODO: test that this will actually catch file changes
                i.check_stat_field('file_pipeline')
            except ValidationError as e:
                errors = e.update_error_dict(errors)
            try:
                i.full_clean()
            except ValidationError as e:
                errors = e.update_error_dict(errors)

        if errors:
            raise ValidationError(errors)
        else:
            for i in self.files:
                i.save()

    def status(self, use_cache=True):
        """
        Get extended status info for this job

        Returns a dict mapping Status to further information.  For DONE this is
        the corresponding tracking records, for MISSING it is the list of
        missing files, for WAITING the list of jobs that have to go before
        this one.  For READY the info is None.

        DONE, WAITING, and MISSING can all occur together, but READY occurs
        only in the absense of the others.
        """
        if self._status is None or not use_cache:
            state = {}
            if self.tracking is None:
                # Get the tracking instance, optimize for case of self.subject
                # being member in queryset called via
                # prefetch_related('tracking'), so no extra DB query for small
                # cost of iterating over the subjects' tracking instances.
                # TODO: verify that this is not just duplicating the similar
                # work (and DB query) in __init__()
                for i in self.subject.tracking.all():
                    if i.flag == self.flag.value:
                        self.tracking = i
                        break

            if self.tracking and self.tracking.id is not None:
                state[Status.DONE] = self.tracking

            waiting = [i for i in self.after if not i.is_done()]
            if waiting:
                state[Status.WAITING] = waiting

            missing = [
                i for i in self.files
                # FIXME? this does not detect broken symlinks
                if not i.file_pipeline.name
                or not i.file_pipeline.storage.exists(i.file_pipeline.name)
            ]
            if missing:
                state[Status.MISSING] = missing

            if not state:
                # we're ready for this job
                state[Status.READY] = None
            self._status = state
        return self._status

    def is_ready(self, use_cache=True):
        return Status.READY in self.status(use_cache=use_cache)

    def is_done(self, use_cache=True):
        return Status.DONE in self.status(use_cache=use_cache)

    def fake_run(self):
        """
        Like __call__() but skips running the job's run function.

        Use with care!
        """
        real_run = self.run
        self.run = lambda *args, **kw: None
        try:
            self()
        finally:
            self.run = real_run

    def fake_undo(self):
        """
        Like run_undo() but skips actually running the job's undo function.

        This will even run if the job's undo attribute is not set.  Use with
        care!
        """
        real_undo = self.undo
        self.undo = lambda *args, **kw: None
        try:
            return self.run_undo()
        finally:
            self.undo = real_undo


class DatasetJob(BaseJob):
    model = get_dataset_model()
    sample_types = None

    def __init__(self, subject, tracking=None):
        if not isinstance(subject, self.model):
            raise ValueError(f'subject must be instance of {self.model}: {subject}')
        super().__init__(subject, tracking)

    @classmethod
    def for_subject(cls, subject, tracking=None):
        """
        Get the job instance for given subject (a.k.a. object).

        Use this in place of the regular constructor to ensure that jobs are
        singleton per subject.

        The given subject may also be a compatible SeqSample, but then will be
        substituted by the sample's dataset.
        """
        if isinstance(subject, SeqSample):
            if cls.is_compatible(subject):
                subject = subject.parent.dataset
            else:
                raise ValueError('incompatible SeqSample instance')

        return super().for_subject(subject, tracking=tracking)

    @classmethod
    def is_compatible(cls, subject):
        if isinstance(subject, SeqSample) and cls.sample_types is not None:
            return subject.sample_type in cls.sample_types
        else:
            return isinstance(subject, get_dataset_model())


class SeqSampleJob(BaseJob):
    model = SeqSample
    sample_types = None

    @classmethod
    def is_compatible(cls, subject):
        """ tell if job is compatible with subject """
        if not isinstance(subject, SeqSample):
            return False
        elif cls.sample_types:
            return subject.sample_type in cls.sample_types
        else:
            return True


class Registry:
    def __init__(self):
        self.jobs = {}

    def clear_cache(self):
        for cls in self.jobs.values():
            cls.clear_cache()

    def register_from_module(self, module):
        if isinstance(module, str):
            # fully dotted module name
            if module not in sys.modules:
                # not yet imported
                import_module(module)
            module_name = module
        else:
            # module imported already
            module_name = module.__name__

        new_jobs = dict(inspect.getmembers(
            sys.modules[module],
            lambda x: inspect.isclass(x)
            and x.__module__ == module_name
            and issubclass(x, BaseJob)
            and x is not BaseJob
            and x.enabled  # FIXME: maybe disable later
        ))

        jobs = self.jobs.copy()
        for name in new_jobs:
            if name in jobs:
                raise RuntimeError(
                    f'{module_name}: job class {name} already exists'
                )
        jobs.update(new_jobs)

        # replace names with classes
        for name, cls in jobs.items():
            if cls._jobs is None:
                cls._jobs = {}  # separate dict per class

            for list_attr in ['after', 'before']:
                lst = getattr(cls, list_attr)
                if lst is None:
                    # replace Nones with distinct lists
                    lst = []
                    setattr(cls, list_attr, lst)
                for i in range(len(lst)):
                    if isinstance(lst[i], str):
                        if lst[i] in jobs:
                            lst[i] = jobs[lst[i]]
                        else:
                            raise ValueError(
                                f'{cls}.{list_attr}: no Job class with name '
                                f'{lst[i]}'
                            )

        # do before-after symmetric closure
        for name, cls in jobs.items():
            for i in cls.after:
                if cls not in i.before:
                    i.before.append(cls)
            for i in cls.before:
                if cls not in i.after:
                    i.after.append(cls)

        # sort by dependency
        sorter = TopologicalSorter()
        names = {job: name for name, job in jobs.items()}
        for name, job in jobs.items():
            sorter.add(name, *(names[dep] for dep in job.after))

        self.jobs = {name: jobs[name] for name in sorter.static_order()}


registry = Registry()
