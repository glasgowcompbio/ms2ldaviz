from __future__ import unicode_literals

from django.db import models
from basicviz.models import Document, Experiment
from basicviz.constants import EXPERIMENT_STATUS_CODE

class Sample(models.Model):
    name = models.CharField(max_length=32)
    experiment = models.ForeignKey(Experiment)

class DocSampleIntensity(models.Model):
    sample = models.ForeignKey(Sample)
    document = models.ForeignKey(Document)
    intensity = models.FloatField(null=True)

class Analysis(models.Model):
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    experiment = models.ForeignKey(Experiment)
    group1 = models.CharField(max_length=2048)
    group2 = models.CharField(max_length=2048)
    ready_code, _ = EXPERIMENT_STATUS_CODE[1]
    status = models.CharField(max_length=128, choices=EXPERIMENT_STATUS_CODE,
                              null=True, default=ready_code)

class AnalysisResult(models.Model):
    analysis = models.ForeignKey(Analysis)
    document = models.ForeignKey(Document)
    pValue = models.FloatField(null=True)
    foldChange = models.FloatField(null=True)
