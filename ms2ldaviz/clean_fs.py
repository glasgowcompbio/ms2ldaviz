# clean the feature set
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

from basicviz.models import *

fs = BVFeatureSet.objects.get(name = 'binned_005')

features = Feature.objects.filter(featureset = fs)

names = {}

for f in features:
	if not f.name in names:
		names[f.name] = []
	names[f.name].append(f)

print len(features)
print len(names)

todelete = {}

for name,subfs in names.items():
	if len(subfs) > 1:
		ndeleted = 0
		for f in subfs:
			a = len(f.featureinstance_set.all())
			b = len(f.mass2motifinstance_set.all())
			if a == 0 and b == 0:
				todelete[f] = True
				ndeleted += 1
		if ndeleted == 0:
			print "Problem: ",name