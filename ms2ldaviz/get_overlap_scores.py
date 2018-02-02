# a script to get the overlap scores for a particular experiment
import sys,os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import *

import csv

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	output_name = sys.argv[2]

	experiment = Experiment.objects.get(name = experiment_name)
	motifs = Mass2Motif.objects.filter(experiment = experiment)

	docm2m = DocumentMass2Motif.objects.filter(mass2motif__in = motifs)

	with open(output_name,'w') as f:
		writer = csv.writer(f)
		for dm in docm2m:
			writer.writerow([dm.document,dm.mass2motif,dm.probability,dm.overlap_score])
