import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import MultiFileExperiment,PeakSet

if __name__ == '__main__':
	mfe_name = sys.argv[1]
	mfe = MultiFileExperiment.objects.get(name = mfe_name)

	peaksets = mfe.peakset_set.all().delete()