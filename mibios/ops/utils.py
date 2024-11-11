from contextlib import ContextDecorator
import cProfile
import logging
import pstats
import tracemalloc

from django.conf import settings
from django.contrib.auth import backends


log = logging.getLogger(__name__)


class RemoteUserBackend(backends.RemoteUserBackend):
    create_unknown_user = False

    def clean_username(self, username):
        """
        Allow server admin to pretend to be a different user
        """
        try:
            real, assumed = settings.ASSUME_IDENTITY
        except Exception:
            pass
        else:
            if username == real:
                log.debug(
                    'Setting REMOTE_USER from {} to {}'.format(real, assumed)
                )
                username = assumed

        return username


class RemoteUserInjection:
    """
    Middleware to manipulate the REMOTE_USER variable on incoming requests

    Do not use this in production.  For development, we most commonly want to
    an authorized user to be logged in when testing views etc. and thus prevent
    the authentication framework to give us a login with the anonymous user,
    who shouldn't be able to access those views.  This middleware should run at
    the earliest point.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.
        """
        Set REMOTE_USER if configured
        """
        try:
            _, assumed = settings.ASSUME_IDENTITY
        except Exception:
            pass
        else:
            log.debug('Setting REMOTE_USER to {}'.format(assumed))
            request.META['REMOTE_USER'] = assumed

        return self.get_response(request)


class Profiling:
    """
    Profiling middleware
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        prof = cProfile.Profile()
        prof.enable()

        response = self.get_response(request)

        prof.disable()
        for sortby in ['cumulative', 'tottime']:
            with open(f'profile.request.{sortby}.txt', 'w') as f:
                ps = pstats.Stats(prof, stream=f).sort_stats(sortby)
                ps.print_stats()

        return response


class profile_this(ContextDecorator):
    """
    Convenience profiling helper

    Use this via with block or decorate a function with it.

    E.g.:

        @profile_this(True):
        def some_func(a, b, c):
            ...

        with profile_this(name='foo_code'):
            ...
    """
    def __init__(self, enabled=True, name=None):
        self.name = name
        self.prof = cProfile.Profile() if enabled else None

    def __call__(self, func):
        # if used as decorator with no given name, we default to use the
        # function name to use for the output file name.
        if self.name is None:
            self.name = func.__name__
        return super().__call__(func)

    def __enter__(self):
        if self.prof is not None:
            self.prof.enable()

    def __exit__(self, exc_type, exc, exc_tb):
        if self.prof is not None:
            self.prof.disable()
            if self.name:
                name = f'{self.name}.'
            else:
                name = ''
            for sortby in ['cumulative', 'tottime']:
                with open(f'profile.{name}{sortby}.txt', 'w') as f:
                    ps = pstats.Stats(self.prof, stream=f).sort_stats(sortby)
                    ps.print_stats()


class TraceMalloc:
    """
    Memory allocation tracing middleware
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.request_count = 0

    def __call__(self, request):
        try:
            a = tracemalloc.take_snapshot()
        except RuntimeError:
            tracemalloc.start()
            a = tracemalloc.take_snapshot()

        response = self.get_response(request)

        b = tracemalloc.take_snapshot()
        stats = b.compare_to(a, 'lineno')
        self.request_count += 1
        path = request.META['PATH_INFO']
        qstr = request.META['QUERY_STRING']
        total_size = sum((i.size for i in stats))
        total_diff = sum((i.size_diff for i in stats))
        with open(f'/tmp/mem_trace.{self.request_count}.txt', 'w') as ofile:
            ofile.write(f'{total_size}\t{total_diff}\t{path}?{qstr}\n')
            for i in stats[:1000]:
                ofile.write(f'{i}\n')

        return response
