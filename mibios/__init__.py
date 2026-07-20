try:
    # _version is expected to be provided by the build system
    from mibios._version import version as __version__
except ImportError:
    # get a version for development settings
    cmd = 'git describe --dirty --tags --long --match "v[0-9]*" --always'
    try:
        import subprocess
        p = subprocess.run(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        __version__ = p.stdout.decode().split(maxsplit=1)[0]
    except Exception:
        __version__ = '_unknown_'


# constants used in TableView and forms, e.g. keyword recognized in the URL
# querystring, declared here to avoid circular imports
QUERY_FILTER = 'filter'
QUERY_EXCLUDE = 'exclude'
QUERY_NEGATE = 'inverse'
QUERY_SHOW = 'show'
QUERY_FORMAT = 'format'
QUERY_AVG_BY = 'avg-by'
QUERY_COUNT = 'count'
QUERY_SEARCH = 'search'
QUERY_Q = 'q'


_registry = None


def get_registry():
    if _registry is None:
        raise RuntimeError(
            "Registry is not yet set up.  It's only available after Django's "
            "apps are set up, i.e. after mibios' app config's ready() has "
            "returned."
        )
    return _registry
