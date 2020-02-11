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
    featureset = models.ForeignKey(BVFeatureSet,null = True,blank=True, on_delete=models.CASCADE)
    metadata = models.TextField(null = True)
    owner = models.ForeignKey(User,null = True, on_delete=models.CASCADE)
    def __str__(self):
        ch = []
        if self.featureset:
            ch.append(str(self.featureset))
        try:
            md = jsonpickle.decode(self.metadata)
            ap = md['Analysis_Polarity']
            if ap:
                ch.append(ap)
            sample_type = md['Sample_type']
            if sample_type:
                ch.append(sample_type)
        except:
            pass
        return self.name + " ({})".format(", ".join(ch))
    def __repr__(self):
        return self.name

class MDBMotif(Mass2Motif):
    motif_set = models.ForeignKey(MDBMotifSet, on_delete=models.CASCADE)
    def save(self,*args,**kwargs):
        super(MDBMotif,self).save(*args,**kwargs)

    def get_comment(self):
        md = jsonpickle.decode(super(MDBMotif,self).metadata)
        return md.get('comment',None)

    comment = property(get_comment)
    
