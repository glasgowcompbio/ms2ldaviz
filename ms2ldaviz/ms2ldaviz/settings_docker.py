from settings import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '^(2&40trp_*ei%$_*p-k598#hu3-w(@9%&&dr&#0##dpag=c%+'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['*']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'django',
        'USER': 'django',
        'PASSWORD': 'j7z3rL40w9',
        'HOST': 'db',
        'PORT': '',
    }
}

CELERY_BROKER_URL = "redis://redis:6379/0"

CHEMSPIDER_APIKEY='fLazuvY8CEiRgF8QfhgrGCr7v1xqD8f9'
