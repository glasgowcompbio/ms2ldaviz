import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import pickle
from annotation.models import *
from basicviz.models import *

if __name__ == '__main__':
	in_name = sys.argv[1]
	term_type = sys.argv[2]
	experiment_name = sys.argv[3]

	assert term_type == 'sub' or term_type == 'taxa'

	with open(in_name,'r') as f:
		export_dict = pickle.load(f)

	experiment = Experiment.objects.get(name = experiment_name)

	print(experiment)

	term_list = export_dict['term_list']
	term_probs = export_dict['term_probs']

	if term_type == 'sub':
		current_values = SubstituentInstance.objects.filter(motif__experiment = experiment)
		all_terms = SubstituentTerm.objects.all()
	else:
		current_values = TaxaInstance.objects.filter(motif__experiment = experiment)
		all_terms = TaxaTerm.objects.all()

	for c in current_values:
		c.delete()

	term_dict = {}
	for term in all_terms:
		term_dict[term.name] = term

	for term_name in term_list:
		if not term_name in term_dict:
			if term_type == 'sub':
				new_term = SubstituentTerm.objects.create(name = term_name)
			else:
				new_term = TaxaTerm.objects.create(name = term_name)
			term_dict[term_name] = new_term

	



	for motif_name,probs in term_probs.items():
		print(motif_name)
		motif = Mass2Motif.objects.get(name = motif_name,experiment = experiment)
		for i,p in enumerate(probs):
			term_name = term_list[i]
			term = term_dict[term_name]
			if term_type == 'sub':
				SubstituentInstance.objects.create(motif = motif,subterm = term,probability = probs[i])
			else:
				TaxaInstance.objects.create(motif = motif,taxterm = term,probability = probs[i])


