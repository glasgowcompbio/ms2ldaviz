from settings import *

# DEBUG = True

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
#     }
# }

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'simon',
        'USER': 'simon',
        'PASSWORD': 'ms2lda',
        'HOST': 'localhost',
        'PORT': '',
    }
}