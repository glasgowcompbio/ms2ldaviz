import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_joe")

import django
django.setup()

from basicviz.models import *

BVFeatureSet.objects.get_or_create(name = 'binned_005')

