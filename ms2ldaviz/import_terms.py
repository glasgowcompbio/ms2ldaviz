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
	term_type = sys.argv[2]
	csv_file = sys.argv[3]
	p_thresh = float(sys.argv[4])

	try:
		experiment = Experiment.objects.get(name = experiment_name)
	except:
		print "Experiment with name {} not found".format(experiment_name)
		sys.exit(0)

	if (not term_type == 'taxa') and (not term_type == 'substituent'):
		print "Unrecognized term type: {}".format(term_type)
		sys.exit(0)

	with open(csv_file,'r') as f:
		reader = csv.reader(f,dialect = 'excel')
		heads = reader.next()
		motif_names = heads[1:]
		motifs = []
		try:
			for motif_name in motif_names:
				motifs.append(Mass2Motif.objects.get(name = motif_name,experiment = experiment))
		except:
			print "Unable to load load motifs from the database"
			sys.exit(0)

		for row in reader:
			term = row[0]
			probabilities = [float(v) for v in row[1:]]
			if term_type == 'taxa':
				term_object = TaxaTerm.objects.get_or_create(name = term)[0]
			elif term_type == 'substituent':
				term_object = SubstituentTerm.objects.get_or_create(name = term)[0]
			print term_object

			for i,motif in enumerate(motifs):
				probability = probabilities[i]
				if probability >= p_thresh:
					if term_type == 'taxa':
						# This overwrites any previous record linking this motif and this term
						tti = TaxaInstance.objects.get_or_create(taxterm = term_object,motif = motif)[0]
						tti.probability = probability
						tti.save()
					elif term_type == 'substituent':
						sti = SubstituentInstance.objects.get_or_create(subterm = term_object,motif = motif)[0]
						sti.probability = probability
						sti.save()




