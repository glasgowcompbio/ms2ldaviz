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

from basicviz.models import Experiment,Mass2Motif,DocumentMass2Motif,MultiFileExperiment

if __name__ == '__main__':
	experiments = Experiment.objects.all()
	for experiment in experiments:
		if len(experiment.multilink_set.all()) == 0:
			print "{}".format(experiment.name)
			mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
			if not mass2motifs:
				print "\t No Mass2Motifs in experiment"
			else:
				docm2m = DocumentMass2Motif.objects.filter(mass2motif = mass2motifs[0])
				if not docm2m[0].overlap_score:
					print "\t Overlaps not computed"
				else:
					print "\t Overlaps computed"
	mfes = MultiFileExperiment.objects.all()
	for mfe in mfes:
		print "{} (multifile)".format(mfe.name)
		links = mfe.multilink_set.all()
		experiment = links[0].experiment
		mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
		if not mass2motifs:
			print "\t No Mass2Motifs in experiment"
		else:
			docm2m = DocumentMass2Motif.objects.filter(mass2motif = mass2motifs[0])
			if not docm2m[0].overlap_score:
				print "\t Overlaps not computed"
			else:
				print "\t Overlaps computed"

