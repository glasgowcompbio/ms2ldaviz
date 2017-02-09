import pickle
import sys
import jsonpickle
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

from basicviz.models import Experiment,Mass2Motif

import django
django.setup()

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	input_file = sys.argv[2]
	with open(input_file,'r') as f:
		input_dict = pickle.load(f)
	motif_metadata = input_dict['motif_metadata']

	experiment = Experiment.objects.get(name = experiment_name)

	for motif_name in motif_metadata:
		motif = Mass2Motif.objects.get(experiment = experiment,name = motif_name)
		motif.metadata = jsonpickle.encode(motif_metadata[motif_name])
		motif.save()