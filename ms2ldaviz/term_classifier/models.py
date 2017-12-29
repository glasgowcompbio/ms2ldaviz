from __future__ import unicode_literals

from django.db import models

from annotation.models import SubstituentTerm
from basicviz.models import Experiment
# Create your models here.

class SubClassifier(models.Model):
	term = models.ForeignKey(SubstituentTerm)
	classifier_type = models.CharField(max_length = 124,null = True)
	classifier = models.TextField(null = True)
	feature_index = models.TextField(null = True)
	experiment = models.ForeignKey(Experiment,null = True)