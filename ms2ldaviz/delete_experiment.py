import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment

if __name__ == '__main__':
	expriment_name = sys.argv[1]
	experiment = Experiment.objects.get(name = expriment_name)
	experiment.delete()