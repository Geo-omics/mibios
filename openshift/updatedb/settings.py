"""
Django settings for the openshift-deployed glarm db updater
"""
from mibios.glamr.settings import *


# Set to True for development but never in production deployment
DEBUG = False

# Add additional apps here:
INSTALLED_APPS.append('django_extensions')

# List of contacts for site adminitrators
ADMINS = [("Robert", "heinro@umich.edu")]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'glamr',
        'USER': 'glamr_django',
        'HOST': 'database.gdick-web-app.svc.cluster.local',
        'PORT': '5432',
    },
}
