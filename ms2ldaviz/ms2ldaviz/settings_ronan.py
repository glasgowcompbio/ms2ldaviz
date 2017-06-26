from settings import *

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'ronan',
        'USER': 'ronan',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '',
    }
}
