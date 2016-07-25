import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Document

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	dict_file = sys.argv[2]
	merge = False
	if 'merge' in sys.argv:
		merge = True
	# Add some things to the document metadata for the beer3 experiment
	experiment = Experiment.objects.get(name = experiment_name)

	with open(dict_file,'r') as f:
		lda_dict = pickle.load(f)
	print "Loaded LDA dictionary from {}".format(dict_file)

	documents = Document.objects.filter(experiment = experiment)
	for document in documents:
		if document.name in lda_dict['doc_metadata']:
			current_metadata = jsonpickle.decode(document.metadata)
			for field in lda_dict['doc_metadata'][document.name]:
				if merge:
					if not field in current_metadata:
						print "Adding {} ({}) to {}".format(field,lda_dict['doc_metadata'][document.name][field],document.name)
						current_metadata[field] = lda_dict['doc_metadata'][document.name][field]
				else:
						print "Adding {} ({}) to {}".format(field,lda_dict['doc_metadata'][document.name][field],document.name)
						current_metadata[field] = lda_dict['doc_metadata'][document.name][field]					
				# if field == 'standard_mol':
				# 	current_metadata['annotation'] = lda_dict['doc_metadata'][document.name][field]
			document.metadata = jsonpickle.encode(current_metadata)
			document.save()

