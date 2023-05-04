"""The project package for mibios"""
import os
from pathlib import Path
import sys

from django.core.management.utils import get_random_secret_key


VAR = 'DJANGO_SETTINGS_MODULE'
DEFAULT_SETTINGS = 'mibios.ops.settings'
LOCAL_SETTINGS = 'settings'


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

    Example usage in settings.py:
        SECRET_KEY = get_secret_key('./secret-key.txt')
    """
    keyfile = Path(keyfile)
    try:
        if not keyfile.exists():
            if keyfile.is_symlink():
                # A broken symlink, not doing anything about that, even though
                # write_text() could create the target just fine, touch()
                # however is a bit touchy, and will, with exits_ok=False raise
                # a "File exist", but create the target with exist_ok=True.
                # Let's just assume that the broken symlink is fully intended.
                raise RuntimeError('is broken symlink')
            print(f'Creating secret key file {keyfile} ...')
            # while this whole function isn't TOCTOU-safe, at least make the
            # key writing safe from preying eyes of other users:
            try:
                keyfile.touch(mode=0o600, exist_ok=False)
            except Exception as e:
                print(f'WARNING: failed touching {keyfile}: '
                      f'{e.__class__.__name__}: {e}')
                raise
            try:
                keyfile.write_text(get_random_secret_key())
            except Exception as e:
                print(f'WARNING: failed writing to {keyfile}: '
                      f'{e.__class__.__name__}: {e}')
                raise
            try:
                keyfile.chmod(0o400)
            except Exception as e:
                print(f'WARNING: failed mode setting {keyfile}: '
                      f'{e.__class__.__name__}: {e}')
                raise

        try:
            return keyfile.read_text()
        except Exception as e:
            print(f'WARNING: failed reading {keyfile}: '
                  f'{e.__class__.__name__}: {e}')
            raise

    except Exception as e:
        # file permissions?
        # TODO: explore consequences of non-permanent keys
        print(f'WARNING: failed accessing secret key file {keyfile}: {e} -- '
              'will be using a temporary key')
        return get_random_secret_key()
