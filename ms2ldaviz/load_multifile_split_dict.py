import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,MultiFileExperiment,MultiLink


from load_dict_functions import *

if __name__ == '__main__':

	directory_path = sys.argv[1]
	filename = sys.argv[2]
	
	verbose = False
	if 'verbose' in sys.argv:
		verbose = True
	with open(directory_path+filename,'r') as f:
		multi_lda_dict = pickle.load(f)

	multi_lda_name = filename.split('/')[-1].split('.')[0]

	print "Loading {}".format(multi_lda_name)
	mfe = MultiFileExperiment.objects.get_or_create(name = multi_lda_name,description = 'none',status = 'loading')[0]
	n_done = 0
	n_loaded = 0

	for lda_dict_name in multi_lda_dict['individual_lda']:
		experiment_name = lda_dict_name
		individual_filename = directory_path + experiment_name + '.dict'
		with open(individual_filename,'r') as f:
			lda_dict = pickle.load(f)
		print experiment_name
		current_e = Experiment.objects.filter(name = experiment_name)
		if len(current_e) > 0:
			print "Experiment of this name already exists, exiting"
			sys.exit(0)
		experiment = Experiment(name=experiment_name)
		experiment.status = 'loading'
		experiment.save()
		ml = MultiLink.objects.get_or_create(experiment = experiment, multifileexperiment = mfe)
		load_dict(lda_dict,experiment,verbose)

	n_loaded += 1
	mfe.status = 'loaded {} of {}'.format(n_loaded,len(multi_lda_dict['individual_lda']))