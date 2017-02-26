from settings import *

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'joewandy',
        'USER': 'joewandy',
        'PASSWORD': 'joewandy',
        'HOST': 'localhost',
        'PORT': '',
    }
}