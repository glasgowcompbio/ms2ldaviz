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
	experiment = Experiment.objects.get(name = experiment_name)
	documents = Document.objects.filter(experiment = experiment)
	for document in documents:
		md = jsonpickle.decode(document.metadata)
		if 'm/z' in md:
			md['parentmass'] = float(md['m/z'])
		document.metadata = jsonpickle.encode(md)
		document.save()