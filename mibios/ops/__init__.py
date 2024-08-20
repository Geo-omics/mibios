"""The project package for mibios"""
from datetime import datetime
from logging import getLogger
import os
from pathlib import Path
import sys
from time import sleep, time

from django.core.management.utils import get_random_secret_key


VAR = 'DJANGO_SETTINGS_MODULE'
DEFAULT_SETTINGS = 'mibios.ops.settings'
LOCAL_SETTINGS = 'settings'

log = getLogger(__name__)


def manage(settings=None, default_settings=DEFAULT_SETTINGS):
    """
    The original manage.py

    This is also the entry point for the manage script when installed vie
    setuptools.  In this case no argument is supplied and the prodcution
    settings are applied by default.  The usual manage.py script shoudl specify
    the development settings.
    """
    if settings is None:
        if VAR in os.environ:
            # already in env, 2nd priority
            pass
        else:
            if os.path.exists(LOCAL_SETTINGS + '.py'):
                # for local/manual deployment
                settings = LOCAL_SETTINGS
                sys.path = [''] + sys.path
            else:
                # last resort but usually for development
                settings = default_settings
    else:
        # passed as argument, has top priority
        pass

    os.environ.setdefault(VAR, settings)
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


def get_secret_key(keyfile):
    """
    Read key from given file

    If file does not exist, generate a key randomly and store it in the file
    first.

    This should be safe if settings.py gets imported multiple times and/or
    concurrently.  Only at most on file is created and everyone gets the same
    key.

    Example usage in settings.py:
        SECRET_KEY = get_secret_key('./secret-key.txt')
    """
    MIN_AGE = 1
    keyfile = Path(keyfile)

    try:

        try:
            while time() - keyfile.stat().st_mtime < MIN_AGE:
                sleep(MIN_AGE)
        except FileNotFoundError:
            if keyfile.is_symlink():
                # A broken symlink, not doing anything about that, even though
                # write_text() could create the target just fine, touch()
                # however is a bit touchy, and will, with exits_ok=False raise
                # a "File exist", but create the target with exist_ok=True.
                # Let's just assume that the broken symlink is fully intended.
                # E.g. while building a docker image, a valid key file will be
                # provided at pod runtime.  Catch this in outer try.
                raise RuntimeError(f'keyfile is broken link: {keyfile}')

            # while this whole function isn't TOCTOU-safe, at least make the
            # key writing safe from preying eyes of other users:
            try:
                keyfile.touch(mode=0o600, exist_ok=False)
            except FileExistsError:
                # Path.touch() uses open with O_CREAT | O_EXCL so other thread
                # or process got here first, give them time to write to file
                sleep(MIN_AGE)
            except Exception as e:
                log.error(f'ERROR ({__name__}) Failed touching {keyfile}: '
                          f'{e.__class__.__name__}: {e}')
                raise
            else:
                # log.warning() + formatting here and below since django
                # logging is not yet configured, level must be warning or
                # higher to be printed (and will go to stderr)
                log.warning(f'[{datetime.now()}] INFO ({__name__}) Creating '
                            f'secret key file: {keyfile}')
                try:
                    keyfile.write_text(get_random_secret_key())
                except Exception as e:
                    log.error(f'ERROR ({__name__}) Failed writing to '
                              f'{keyfile}: {e.__class__.__name__}: {e}')
                    raise
                try:
                    keyfile.chmod(0o400)
                except Exception as e:
                    log.error(f'ERROR ({__name__}) Failed mode setting '
                              f'{keyfile}: {e.__class__.__name__}: {e}')
                    raise
        except Exception as e:
            log.error(f'ERROR ({__name__}) Failed getting age of {keyfile}: '
                      f'{e.__class__.__name__}: {e}')
            sleep(MIN_AGE)

    except Exception:
        pass

    try:
        return keyfile.read_text()
    except Exception as e:
        # file permissions?
        # TODO: explore consequences of non-permanent keys
        log.warning(f'Failed accessing secret key file {keyfile}: {e} -- '
                    f'will be using a temporary key')
        return get_random_secret_key()
