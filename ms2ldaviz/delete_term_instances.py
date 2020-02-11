import os
import pickle
import numpy as np
import sys
import csv
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Mass2Motif
from annotation.models import TaxaTerm,SubstituentTerm,TaxaInstance,SubstituentInstance

if __name__ == '__main__':
	experiment_name = sys.argv[1]

	try:
		experiment = Experiment.objects.get(name = experiment_name)
	except:
		print("Experiment with name {} not found".format(experiment_name))
		sys.exit(0)


	motifs = Mass2Motif.objects.filter(experiment = experiment)
	ti = TaxaInstance.objects.filter(motif__in = motifs)
	for t in ti:
		t.delete()

	si = SubstituentInstance.objects.filter(motif__in = motifs)
	for s in si:
		s.delete()
