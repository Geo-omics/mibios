""" helpers to generate and serve model graphs """
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command

from mibios import get_registry

STATIC_SUB_DIR = 'model_graphs'


def get_app_names():
    """
    Get names for app for which we want model graphs
    """
    return [i for i in get_registry().apps.keys() if i != 'mibios']


def get_image_path(app_name):
    return Path(STATIC_SUB_DIR) / f'{app_name}.png'


def get_model_graph_info():
    """ get a map mapping each app name to the corresponding image path """
    return {
        app_name: get_image_path(app_name)
        for app_name in get_app_names()
    }


def make_graph(app_name, output_dir=None):
    """
    Create and save a graph

    Will raise FileNotFoundError if a non-existing output_dir is given.  If no
    output_dir is provided, then the image files will be saved under the static
    root in a separate subdirectory which will be created if needed.  The
    static root directory itself must exist.  May raise other exceptions unless
    filesystem interactions go as intended.  Existing files will be
    overwritten.

    Depends on the django_extensions app being properly installed.
    """
    if 'django_extensions' not in settings.INSTALLED_APPS:
        raise ImproperlyConfigured(
            "You need to add 'django_extensions' to INSTALLED_APPS"
        )

    if output_dir is None:
        image_path = Path(settings.STATIC_ROOT) / get_image_path(app_name)
        # create the STATIC_SUB_DIR if needed, but STATIC_ROOT must exist
        image_path.parent.mkdir(exist_ok=True)
    else:
        image_path = Path(output_dir) / (get_image_path(app_name).name)

    call_command(
        'graph_models',
        app_name,
        output=str(image_path),
        exclude_models=['Model'],
        no_inheritance=True,
    )
    return image_path


def make_all_graphs(output_dir=None):
    images = []
    for app_name, image_path in get_model_graph_info().items():
        img = make_graph(app_name, output_dir=output_dir)
        images.append(img)
    return images
