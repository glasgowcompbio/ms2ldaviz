import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import numpy as np
import bisect
import jsonpickle
from basicviz.models import Experiment,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,Alpha,Document,BVFeatureSet

if __name__ == '__main__':
	es = Experiment.objects.all()
	for e in es:
		docs = Document.objects.filter(experiment = e)
		doc = docs[0]
		f = FeatureInstance.objects.filter(document = doc)[0].feature
		fs = f.featureset
		if fs:
			e.featureset = fs
			e.save()
		
