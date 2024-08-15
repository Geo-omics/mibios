import cProfile
from functools import wraps
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
            with open(f'profile.{sortby}.txt', 'w') as f:
                ps = pstats.Stats(prof, stream=f).sort_stats(sortby)
                ps.print_stats()

        return response


def profile(func):
    """
    Profile given function or method
    """
    @wraps(func)
    def profiled_func(*args, **kwargs):
        prof = cProfile.Profile()
        prof.enable()

        retval = func(*args, **kwargs)

        prof.disable()
        for sortby in ['cumulative', 'tottime']:
            with open(f'profile.{func.__name__}.{sortby}.txt', 'w') as f:
                ps = pstats.Stats(prof, stream=f).sort_stats(sortby)
                ps.print_stats()

        return retval

    return profiled_func


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
