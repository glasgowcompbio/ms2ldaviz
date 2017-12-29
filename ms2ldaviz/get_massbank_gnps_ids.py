# script to get the DB IDs for massbank and GNPS docs and motifs
import os,csv
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import Experiment,Mass2Motif,Document

def get_doc_ids(experiment):
	docs = Document.objects.filter(experiment = experiment)
	doc_dict = {}
	for doc in docs:
		doc_dict[doc.name] = doc.id
	return doc_dict


def get_motif_ids(experiment):
	docs = Mass2Motif.objects.filter(experiment = experiment)
	doc_dict = {}
	for doc in docs:
		doc_dict[doc.name] = doc.id
	return doc_dict


def write_dict(d,filename):
	with open(filename,'w') as f:
		writer = csv.writer(f,dialect='excel')
		for k,v in d.items():
			writer.writerow([k,v])

if __name__ == '__main__':
	experiment = Experiment.objects.get(name = 'massbank_binned_005')
	mb_docs = get_doc_ids(experiment)
	mb_motifs = get_motif_ids(experiment)
	experiment = Experiment.objects.get(name = 'gnps_binned_005')
	gn_docs = get_doc_ids(experiment)
	gn_motifs = get_motif_ids(experiment)

	write_dict(mb_docs,'mb_doc_id.csv')
	write_dict(mb_motifs,'mb_motif_id.csv')
	write_dict(gn_docs,'gn_doc_id.csv')
	write_dict(gn_motifs,'gn_motif_id.csv')


