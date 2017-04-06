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

class Analysis(models.Model):
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    experiment = models.ForeignKey(Experiment)
    group1 = models.CharField(max_length=256)
    group2 = models.CharField(max_length=256)

class AnalysisResult(models.Model):
    analysis = models.ForeignKey(Analysis)
    document = models.ForeignKey(Document)
    pValue = models.FloatField(null=True)
    foldChange = models.FloatField(null=True)
