from enum import Enum
from importlib import import_module
import inspect
import sys

from django.utils.functional import cached_property

from .models import File, SampleTracking


Status = Enum('Status', ['READY', 'DONE', 'WAITING', 'MISSING'])


class Job:
    enabled = True
    flag = None
    after = None
    before = None
    sample_types = None
    required_files = None

    def __init__(self, sample):
        """
        don't use this constructor directly, use for_sample() instead.
        """
        if not self.is_compatible(sample):
            raise ValueError(
                f'{sample}: {type(self)} is not for sample type '
                f'{sample.sample_type}'
            )
        self.sample = sample

    _jobs = None
    """ object holder for for_sample(); to be initialized to {} in registry """

    @classmethod
    def for_sample(cls, sample):
        """
        Get the job instance for given sample

        Use this in place of the regular constructor to ensure that jobs are
        singleton per sample.
        """
        if sample not in cls._jobs:
            obj = cls(sample)
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
    def status(self):
        state = {}
        try:
            track = SampleTracking.objects.get(
                flag=self.flag,
                sample=self.sample,
            )
        except SampleTracking.DoesNotExist:
            pass
        else:
            state[Status.DONE] = track

        waiting = [i for i in self.after if not i.is_done()]
        if waiting:
            state[self.WAITING] = waiting

        if self.required_files:
            ftypes = (File.objects
                      .filter(filetype__in=self.required_files,
                              sample=self.sample)
                      .values_list('filetype', flat=True))
            missing = [i for i in self.required_files if i not in ftypes]
            if missing:
                state[Status.MISSING] = missing

        if not state:
            # we're ready for this step
            state[Status.READY] = None
        return state

    def __str__(self):
        return f'{self.sample.sample_id}@{self.flag}'

    def is_ready(self):
        return Status.READY in self.status

    def is_done(self):
        return Status.DONE in self.status


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
