import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import numpy as np
import bisect
import jsonpickle
from basicviz.models import Experiment,Feature,Mass2Motif,Mass2MotifInstance,Alpha,Document,BVFeatureSet

if __name__ == '__main__':
    bfs = BVFeatureSet.objects.get(name = 'binned_005')
    features = Feature.objects.filter(featureset = bfs)
    frags = [f for f in features if f.name.startswith('fragment') and len(f.name.split('.')[-1])>4]
    print frags[1].name
    # loss = [f for f in features if f.name.startswith('loss')]
    print len(frags)
    