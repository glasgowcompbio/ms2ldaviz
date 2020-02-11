import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

import sys
import csv

from basicviz.models import Experiment,Mass2Motif
from annotation.models import SubstituentTerm,TaxaTerm,SubstituentInstance,TaxaInstance

if __name__=='__main__':
	experiment_name = sys.argv[1]
	term_type = sys.argv[2]
	file_name = sys.argv[3]


	term_input = []


	experiment = Experiment.objects.get(name = experiment_name)
	motifs = Mass2Motif.objects.filter(experiment = experiment)
	if term_type == 'sub':
		current_terms = SubstituentInstance.objects.filter(motif__in = motifs)
	else:
		current_terms = TaxaInstance.objects.filter(motif__in = motifs)

	print("Deleting {} term instances".format(len(current_terms)))
	for term in current_terms:
		term.delete()

	with open(file_name,'r') as f:
		reader = csv.reader(f)
		for line in reader:
			term_input.append(line)
	
	for motif_name,term_name,probability in term_input:
		motif = Mass2Motif.objects.get(experiment = experiment,name = motif_name)
		if term_type == 'sub':
			term = SubstituentTerm.objects.get_or_create(name = term_name)[0]
			ti = SubstituentInstance.objects.create(subterm = term,motif = motif,probability = probability)
		else:
			term = TaxaTerm.objects.get_or_create(name = term_name)[0]
			ti = TaxaInstance.objects.create(taxterm = term,motif = motif,probability = probability)