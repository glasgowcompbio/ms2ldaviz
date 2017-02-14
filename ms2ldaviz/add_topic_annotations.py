# Disposable script for adding an annotation term to a topic which has a common name
import sys
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Mass2Motif

if __name__ == '__main__':
	experiment_name = sys.argv[1]
	experiment = Experiment.objects.get(name = experiment_name)
	motifs = Mass2Motif.objects.filter(experiment = experiment)

	for motif in motifs:
		metadata = jsonpickle.decode(motif.metadata)
		if 'common_name' in metadata:
			metadata['annotation'] = metadata['common_name']
		motif.metadata = jsonpickle.encode(metadata)
		motif.save()
		

