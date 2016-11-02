import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


# This is to process files that have document names stored as mz_rt
# it extracts the acutal mz and rt from the name and adds them to the metadata
# This version works for multi-file experiments


import jsonpickle
import csv

from basicviz.models import SystemOptions

if __name__ == '__main__':
	# set the minimum count for peaksets to be included in the heatmaps
	s = SystemOptions.objects.get_or_create(key = 'heatmap_minimum_display_count',experiment = None)[0]
	s.value = '5'
	s.save()


	s = SystemOptions.objects.get_or_create(key = 'peakset_matching_tolerance',experiment = None)[0]
	s.value = '5,5'
	s.save()

	s = SystemOptions.objects.get_or_create(key = 'log_peakset_intensities',experiment = None)[0]
	s.value = 'true'
	s.save()

	s = SystemOptions.objects.get_or_create(key = 'doc_m2m_threshold',experiment = None)[0]
	s.value = '0.05'
	s.save()

	s = SystemOptions.objects.get_or_create(key = 'default_doc_m2m_score',experiment = None)[0]
	s.value = 'probability'
	s.save()