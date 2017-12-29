import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()
from django.db import transaction
import sys
import csv

from basicviz.models import Experiment,Mass2Motif
from annotation.models import SubstituentTerm,TaxaTerm,SubstituentInstance,TaxaInstance

if __name__ == '__main__':
	experiment = sys.argv[1]
	term_type = sys.argv[2]
	term_file = sys.argv[3]

	e = Experiment.objects.get(name = experiment)
	motifs = Mass2Motif.objects.filter(experiment = e)
	motif_dict = {}
	for motif in motifs:
		motif_dict[motif.name] = motif


	with transaction.atomic():
		if term_type == 'sub':
			current_terms = SubstituentTerm.objects.all()
			SubstituentInstance.objects.filter(motif__in = motifs).delete()
		else:
			current_terms = TaxaTerm.objects.all()
			TaxaInstance.objects.filter(motif__in = motifs).delete()

	term_dict = {}
	for term in current_terms:
		term_dict[term.name] = term

	with transaction.atomic():
		with open(term_file,'r') as f:
			reader = csv.reader(f)
			for line in reader:
				motif = line[0]
				term = line[1]
				p = float(line[2])
				z = float(line[3])
				if not term in term_dict:
					if term_type == 'sub':
						this_term = SubstituentTerm.objects.create(name = term)
					else:
						this_term = TaxaTerm.objects.create(name = term)
				else:
					this_term = term_dict[term]

				this_motif = motif_dict[motif]

				if term_type == 'sub':
					SubstituentInstance.objects.create(subterm = this_term,motif = this_motif,probability = p,z_score = z)
				else:
					TaxaInstance.objects.create(taxterm = this_term,motif = this_motif,probability = p,z_score = z)


