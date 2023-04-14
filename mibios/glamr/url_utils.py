"""
URL resolving/reversing utilities
"""
from django.conf import settings
from django.urls import get_script_prefix, URLResolver
from django.urls.resolvers import RegexPattern
from django.utils.translation import get_language


# Making django.urls.reverse() faster: It turns out that when generating many
# (as in thousands) URLs to be put into a template (e.g.
# MapMixin.get_map_points) the calls to obtain the language and prefix are
# surprisingly expensive.  So we below we obtain these values here once
# globally.  Further the fast_reverse function will work with a single global
# URLResolver instance.  Further, while reverse() accepts the urlconf,
# current_app parameters fast_reverse() does not.  fast_reverse() also does not
# handle namespacing in any way.
# This is all quite icky, depending on django-internal implementation details
# and is careless with thread safety.
# On the upside, we get five-fold timing improvement comparing reverse.
# and fast_reverse (about 20 -> 4 microseconds) and saving ~0.3 seconds on the
# glamr frontpage with 6000 URLs for map points.

_language_code = get_language()
# _script_prefix = get_script_prefix()


class FastURLReverser(URLResolver):
    """
    URLResolver replacement for faster reverse URL generation

    This is only here to support fast_reverse
    """
    @property
    def reverse_dict(self):
        if _language_code not in self._reverse_dict:
            self._populate()
        return self._reverse_dict[_language_code]


_resolver = FastURLReverser(RegexPattern(r'^/'), settings.ROOT_URLCONF)


def fast_reverse(viewname, args=None, kwargs=None):
    """
    Faster django.urls.reverse (mostly) drop-in
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return _resolver._reverse_with_prefix(
        viewname, get_script_prefix(), *args, **kwargs
    )
