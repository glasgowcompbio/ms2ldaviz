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
		if e.featureset == None:
			print e
			docs = Document.objects.filter(experiment = e)
			doc = docs[0]
			fl = []
			fl = FeatureInstance.objects.filter(document = doc)
			if len(fl)>0:
				fs = fl[0].feature.featureset
				if fs:
					e.featureset = fs
					e.save()

