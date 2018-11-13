import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import *

BVFeatureSet.objects.get_or_create(name='binned_005')
BVFeatureSet.objects.get_or_create(name='binned_01')
BVFeatureSet.objects.get_or_create(name='binned_05')
BVFeatureSet.objects.get_or_create(name='binned_1')
BVFeatureSet.objects.get_or_create(name='binned_5')
