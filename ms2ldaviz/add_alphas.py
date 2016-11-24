# Script to add the min and max mz values to feature objects
# takes as input the path of a dictionary file that must have a 'features' key

import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


from basicviz.models import Mass2Motif,Experiment,Alpha

if __name__ == '__main__':
	infile = sys.argv[1]
	with open(infile,'r') as f:
		lda_dict = pickle.load(f)
	experiment_name = infile.split('/')[-1].split('.')[0]
	experiment = Experiment.objects.get(name = experiment_name)

	motifs = Mass2Motif.objects.filter(experiment = experiment)
	for motif in motifs:
		topic_pos = topic_pos = lda_dict['topic_index'][motif.name]
		alp = Alpha.objects.get_or_create(mass2motif = motif,value = lda_dict['alpha'][topic_pos])

