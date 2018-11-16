# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

from basicviz.models import BVFeatureSet,Feature

# Create your models here.
class MDBMotifSet(models.Model):
    name = models.CharField(max_length=124,null=False)
    description = models.TextField(null = True)
    featureset = models.ForeignKey(BVFeatureSet,null = True)

class MDBMotif(models.Model):
    name = models.CharField(max_length = 124,null = False)
    motif_set = models.ForeignKey(MDBMotifSet)
    annotation = models.TextField(null = True)
    short_annotation = models.TextField(null = True)
    comment = models.TextField(null = True)
    # todo: add metadata

class MDBMotifInstance(models.Model):
    motif = models.ForeignKey(MDBMotif)
    feature = models.ForeignKey(Feature)
    probability = models.FloatField(null = True)
    
