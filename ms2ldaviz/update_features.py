# Script to add the min and max mz values to feature objects
# takes as input the path of a dictionary file that must have a 'features' key

import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


from basicviz.models import Feature,Experiment

if __name__ == '__main__':
	infile = sys.argv[1]
	with open(infile,'r') as f:
		lda_dict = pickle.load(f)
	experiment_name = infile.split('/')[-1].split('.')[0]
	experiment = Experiment.objects.get(name = experiment_name)

	features = lda_dict['features']

	n_features = len(features)
	ndone = 0
	for feature in features:
		f = Feature.objects.get(name = feature,experiment = experiment)
		f.min_mz = features[feature][0]
		f.max_mz = features[feature][1]
		f.save()
		ndone += 1
		if ndone % 100 == 0:
			print("Done {} of {}".format(ndone,n_features))
