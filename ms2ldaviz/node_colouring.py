import os
import pickle
import csv
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Document


def load_pimp_peaks(filename):
	pimp_peaks = []
	with open (filename) as peak_info:
		peak_reader = csv.reader(peak_info)
		peak_reader.next()
		for row in peak_reader:
			#peak id, peak mass, peak rt, peak logFC
			pimp_peaks.append([int(row[0]), float(row[1]), float(row[2]), float(row[4])])
	return pimp_peaks

def load_ms2lda_peaks(experiment_id):
	experiment = Experiment.objects.get(id = experiment_id)
	documents = Document.objects.filter(experiment = experiment)
	ms2lda_peaks = []
	for doc in documents:
		ms2lda_peaks.append((doc.id, doc.mass, doc.rt))
	return ms2lda_peaks

#still need to delete a peak if there are more than 1 peak matching
#min(myList, key=lambda x:abs(x-myNumber)) ?? for the closest one
def match_peaks(pimp_peaks, ms2lda_peaks):
	matching_peaks = []
	for peak in ms2lda_peaks:
		for pimp_peak in pimp_peaks:
			if (pimp_peak[1]-0.000005 < peak[1] < pimp_peak[1]+0.000005 
				and pimp_peak[2] - 5 < peak[2] < pimp_peak[2] +5):
				matching_peaks.append((peak, pimp_peak[3]))
	return matching_peaks

def add_metadata(experiment_id, matching_peaks):
	experiment = Experiment.objects.get(id = experiment_id)
	documents = Document.objects.filter(experiment = experiment)

	for peak in matching_peaks:
		for doc in documents:			
			if doc.id == peak[0][0]:
				current_metadata = jsonpickle.decode(doc.metadata)
				current_metadata['logFC'] = peak[1]
				doc.metadata = jsonpickle.encode(current_metadata)
				doc.save()

filename = 'peak_info.csv'
experiment_id = 1
pimp_peaks = load_pimp_peaks(filename)
ms2lda_peaks = load_ms2lda_peaks(experiment_id)
matching_peaks = match_peaks(pimp_peaks, ms2lda_peaks)
add_metadata(experiment_id, matching_peaks)
for peaks in matching_peaks:
	print peaks, '\n'
# print('number of matching peaks: ', len(matching_peaks))
print ('Fin')

