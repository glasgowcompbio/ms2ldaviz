from django.db import models

from basicviz.models import Mass2Motif
# Create your models here.

class SubstituentTerm(models.Model):
	name = models.CharField(max_length = 128,null = False,unique = True)

class TaxaTerm(models.Model):
	name = models.CharField(max_length = 128,null = False,unique = True)

class SubstituentInstance(models.Model):
	subterm = models.ForeignKey(SubstituentTerm,null = False)
	motif = models.ForeignKey(Mass2Motif,null = False)
	probability = models.FloatField(null = False)

class TaxaInstance(models.Model):
	taxterm = models.ForeignKey(TaxaTerm,null = False)
	motif = models.ForeignKey(Mass2Motif,null = False)
	probability = models.FloatField(null = False)

