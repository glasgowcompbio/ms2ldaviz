# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import jsonpickle

from django.db import models
from django.contrib.auth.models import User
from basicviz.models import BVFeatureSet,Feature,Mass2Motif

# Create your models here.
class MDBMotifSet(models.Model):
    name = models.CharField(max_length=124,null=False,unique=True)
    description = models.TextField(null = True)
    featureset = models.ForeignKey(BVFeatureSet,null = True,blank=True)
    metadata = models.TextField(null = True)
    owner = models.ForeignKey(User,null = True)
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

class MDBMotif(Mass2Motif):
    motif_set = models.ForeignKey(MDBMotifSet)
    def save(self,*args,**kwargs):
        super(MDBMotif,self).save(*args,**kwargs)

    def get_comment(self):
        md = jsonpickle.decode(super(MDBMotif,self).metadata)
        return md.get('comment',None)

    comment = property(get_comment)
    
