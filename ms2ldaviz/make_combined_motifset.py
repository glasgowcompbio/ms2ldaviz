# scipt to make combined motifset
import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()



combined_filename = '/home/combined_motifs.csv'
import csv
motifs = []
with open(combined_filename,'r') as f:
	reader = csv.reader(f)
	for line in reader:
		motifs.append(line)