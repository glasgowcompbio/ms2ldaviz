# see http://docs.celeryproject.org/en/latest/django/first-steps-with-django.html
# also see http://stackoverflow.com/questions/19926750/django-importerror-cannot-import-name-celery-possible-circular-import

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

app = Celery('proj')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))