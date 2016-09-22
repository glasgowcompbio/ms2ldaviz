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

from basicviz.models import MultiFileExperiment,Experiment,Document,PeakSet,IntensityInstance

if __name__ == '__main__':
	multifile_experiment_name = sys.argv[1]
	combined_file_name = sys.argv[2]

	mfe = MultiFileExperiment.objects.get(name = multifile_experiment_name)
	links = mfe.multilink_set.all()
	print "Found {} links".format(len(links))
	experiments = [l.experiment for l in links]
	
	raw_data = []
	mz_list = []
	rt_list = []
	id_list = []
	intensities = []
	counts = []
	count_hist = {}
	with open(combined_file_name,'r') as f:
		reader = csv.reader(f,delimiter='\t')
		heads = reader.next()
		for line in reader:
			raw_data.append(line)
			id = int(line[0])
			mz = float(line[1])
			rt = float(line[2])
			new_intensity = [float(a) if not a=="" else 'nan' for a in line[3:]]
			new_count = sum([1 for a in new_intensity if type(a) == float])
			counts.append(new_count)
			if new_count in count_hist:
				count_hist[new_count] += 1
			else:
				count_hist[new_count] = 1
 			intensities.append(new_intensity)
 			mz_list.append(mz)
 			rt_list.append(rt)
 			id_list.append(id)

	print "Loaded {} peaksets".format(len(raw_data))
	print "Diagnostics: intensities present:"
	ch = zip(count_hist.keys(),count_hist.values())
	ch = sorted(ch,key = lambda x: x[0])

	for c,h in ch:
		print c,h


	# Hack the names about to make them match - this should be provided to the script
	# to avoid errors...fix!
	experiment_match = {}
	short_heads = []
	for head in heads[3:]:
		s = head.split('_')
		new_name = s[0] + '_' + s[1] + '_' + s[2]
		short_heads.append(new_name)
		match = [e for e in experiments if new_name in e.name]
		experiment_match[new_name] = match[0]

	

	print "Making the peakset objects"
	for i,mz_value in enumerate(mz_list):
		rt_value = rt_list[i]
		new_peakset = PeakSet.objects.get_or_create(multifileexperiment = mfe,mz = mz_value,rt = rt_value,
													original_file = combined_file_name,original_id = id_list[i])[0]
		for j,intense in enumerate(intensities[i]):
			if not intense == 'nan':
				exper = experiment_match[short_heads[j]]
				ii = IntensityInstance.objects.get_or_create(peakset = new_peakset,experiment = exper)[0]
				ii.intensity = intense
				ii.save()
		if i%100 == 0:
			print "Done: {}".format(i)

