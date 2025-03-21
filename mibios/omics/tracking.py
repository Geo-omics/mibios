from enum import Enum
from functools import cached_property
from importlib import import_module
import inspect
import sys

from django.db.transaction import atomic

from .models import SampleTracking


Status = Enum('Status', ['READY', 'DONE', 'WAITING', 'MISSING'])


class Job:
    enabled = True
    flag = None
    after = None
    before = None
    sample_types = None
    required_files = None
    run = None

    def __init__(self, sample, tracking=None):
        """
        don't use this constructor directly, use for_sample() instead.
        """
        if not self.is_compatible(sample):
            raise ValueError(
                f'{sample}: {type(self)} is not for sample type '
                f'{sample.sample_type}'
            )
        self.sample = sample

        if tracking is None:
            if hasattr(sample, '_prefetched_objects_cache'):
                for i in sample._prefetched_objects_cache.get('tracking', []):
                    if i.flag == self.flag.value:
                        tracking = i
                        break
        else:
            if self.flag.value != tracking.flag:
                raise ValueError('flag mismatch')
        self.tracking = tracking

        self._status = None

    _jobs = None
    """ object holder for for_sample(); to be initialized to {} in registry """

    def __str__(self):
        if self.is_done():
            status = Status.DONE.name
        elif self.is_ready():
            status = Status.READY.name
        elif Status.WAITING in self.status:
            status = Status.WAITING.name
        elif Status.MISSING in self.status:
            status = Status.MISSING.name
        else:
            status = self.status()

        return f'{self.sample.sample_id}/{self.__class__.__name__} [{status}]'

    @atomic
    def __call__(self):
        """
        Run the job

        This Calls the callable stored under the run class attribute with the
        sample as positional argument and tracks the success in the sample
        tracking table.

        This will not check certain pre-conditions, e.g. one should call e.g.
        is_ready() beforehand.
        """
        if self.run is None:
            raise RuntimeError('job can not be run')
        if not self.enabled:
            raise RuntimeError('job is disabled')

        kw = {}

        for i in self.files:
            # may raise ValidationError
            i.full_clean()
            i.verify_with_pipeline()

        if len(self.files) == 1:
            kw['file'] = self.files[0].path
        elif len(self.files) > 1:
            # SampleLoadMixin etc only expect (and currently only have need
            # for one file per job) one file with the file kw args
            raise RuntimeError('implementation supports only single file')

        retval = type(self).run(self.sample, **kw)

        for i in self.files:
            # may raise ValidationError
            # TODO: test that this will actually catch file changes
            i.full_clean()
            i.save()

        if self.tracking is None:
            self.tracking = SampleTracking(
                flag=self.flag,
                sample=self.sample,
            )
        self.tracking.full_clean()
        self.tracking.save()

        # clear cached tracking data, then update status
        try:
            del self.sample._prefetched_objects_cache['tracking']
        except (AttributeError, KeyError):
            pass
        self.status(use_cache=False)

        return retval

    @classmethod
    def for_sample(cls, sample, tracking=None):
        """
        Get the job instance for given sample

        Use this in place of the regular constructor to ensure that jobs are
        singleton per sample.
        """
        if sample not in cls._jobs:
            obj = cls(sample, tracking=tracking)
            # store obj now to avoid infinite recursion trying to instatiate
            # before and after jobs.
            cls._jobs[sample] = obj
            # FIXME: job declarations should eventually have some
            # conditionals, e.g. do next job only for some sampletype
            obj.after = [i.for_sample(sample) for i in obj.after
                         if i.is_compatible(sample)]
            obj.before = [i.for_sample(sample) for i in obj.before
                          if i.is_compatible(sample)]
        return cls._jobs[sample]

    @classmethod
    def is_compatible(cls, sample):
        """ tell if job is compatible with sample """
        if cls.sample_types:
            return sample.sample_type in cls.sample_types
        else:
            return True

    @cached_property
    def files(self):
        if self.required_files is None:
            return []

        return [self.sample.get_omics_file(i) for i in self.required_files]

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
                # Get the tracking instance, optimize for case of self.sample
                # being member in queryset called via
                # prefetch_related('tracking'), so no extra DB query for small
                # cost of iterating over the samples' tracking instances.
                # TODO: verify that this is not just duplicating the similar
                # work (and DB query) in __init__()
                for i in self.sample.tracking.all():
                    if i.flag == self.flag.value:
                        self.tracking = i
                        break

            if self.tracking:
                state[Status.DONE] = self.tracking

            waiting = [i for i in self.after if not i.is_done()]
            if waiting:
                state[Status.WAITING] = waiting

            if missing := [i for i in self.files if not i.path.exists()]:
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


class Registry:
    def __init__(self):
        self.jobs = {}

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
            and issubclass(x, Job)
            and x is not Job
            and x.enabled  # FIXME: maybe disable later
        ))

        jobs = self.jobs
        for name in new_jobs:
            if name in jobs:
                raise RuntimeError(
                    f'{module_name}: job class {name} already exists'
                )
        jobs.update(new_jobs)

        # replace names with classed
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

        self.jobs = jobs


registry = Registry()
