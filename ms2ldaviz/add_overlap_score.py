import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


# This is to process files that have document names stored as mz_rt
# it extracts the acutal mz and rt from the name and adds them to the metadata
# This version works for multi-file experiments


import jsonpickle
import csv

from basicviz.models import Experiment,DocumentMass2Motif,Document
from basicviz.views import compute_overlap_score


if __name__ == '__main__':
	experiment_name = sys.argv[1]
	experiment = Experiment.objects.get(name = experiment_name)
	documents = Document.objects.filter(experiment = experiment)
	document_mass2motifs = DocumentMass2Motif.objects.filter(document__in = documents)
	print "Found {} documents, and {} document-mass2motif links".format(len(documents),len(document_mass2motifs))
	n_done = 0
	for d_m2m in document_mass2motifs:
		new_score = compute_overlap_score(d_m2m.mass2motif,d_m2m.document)
		d_m2m.overlap_score = new_score
		d_m2m.save()
		n_done += 1
		if n_done % 500 == 0:
			print "Done {} of {}".format(n_done,len(document_mass2motifs))
