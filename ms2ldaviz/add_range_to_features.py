# Script to add the min and max mz values to feature objects
# takes as input the an experiment name and a dictionary of features and their ranges

import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


from basicviz.models import Feature,Experiment

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	infile = sys.argv[2]
	with open(infile,'r') as f:
		features = pickle.load(f)
	
	experiment = Experiment.objects.get(name = experiment_name)

	n_features = len(features)
	ndone = 0
	for feature in features:
		try:
			f = Feature.objects.get(name = feature,experiment = experiment)
			f.min_mz = features[feature][0]
			f.max_mz = features[feature][1]
			f.save()
			ndone += 1
		except:
			print("feature not found")
		if ndone % 100 == 0:
			print("Done {} of {}".format(ndone,n_features))
