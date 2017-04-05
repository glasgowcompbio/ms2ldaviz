from __future__ import unicode_literals

from django.db import models
from basicviz.models import Document, Experiment

class Sample(models.Model):
    name = models.CharField(max_length=32)
    experiment = models.ForeignKey(Experiment)

class DocSampleIntensity(models.Model):
    sample = models.ForeignKey(Sample)
    document = models.ForeignKey(Document)
    intensity = models.FloatField(null=True)
