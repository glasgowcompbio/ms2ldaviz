import pickle
import numpy as np
import sys
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Mass2Motif

if __name__ == '__main__':
	with open('/Users/simon/git/lda/notebooks/beer3.dict','r') as f:
		lda = pickle.load(f)

	experiment = Experiment.objects.get(name='beer3')
	

	for m2m in lda['topic_metadata']:
		motif = Mass2Motif.objects.get(name = m2m,experiment=experiment)
		motif.metadata = jsonpickle.encode(lda['topic_metadata'][m2m])
		motif.save()	
