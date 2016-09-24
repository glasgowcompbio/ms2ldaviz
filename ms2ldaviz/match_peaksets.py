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


def hit(mass1,mass2,tol):
	if 1e6*abs(mass1-mass2)/mass1 < tol:
		return True
	else:
		return False

if __name__ == '__main__':
	multifile_experiment_name = sys.argv[1]
	
	mfe = MultiFileExperiment.objects.get(name = multifile_experiment_name)
	links = mfe.multilink_set.all()
	print "Found {} links".format(len(links))
	experiments = [l.experiment for l in links]
	
	peaksets = mfe.peakset_set.all().order_by('mz')
	print "Found {} peaksets".format(len(peaksets))

	for experiment in experiments:
		print "performing matching for experiment {}".format(experiment)
		docs = experiment.document_set.all()
		print "\t found {} documents".format(len(docs))
		experiment_mz_rt = []
		for doc in docs:
			md = jsonpickle.decode(doc.metadata)
			mz = md['mz']
			rt = md['rt']
			experiment_mz_rt.append((doc,mz,rt))
		
		experiment_mz_rt = sorted(experiment_mz_rt,key = lambda x: x[1])

		n_done = 0
		low_pos = 0
		high_pos = 0

		mass_tol = 5 # ppm
		rt_tol = 2 # seconds?
		peaksets_found = [0 for p in peaksets]
		for doc,mz,rt in experiment_mz_rt:
			while not hit(mz,peaksets[low_pos].mz,mass_tol) and peaksets[low_pos].mz < mz:
				low_pos += 1
				if low_pos >= len(peaksets):
					low_pos -= 1
					break
			high_pos = low_pos + 1
			if high_pos >= len(peaksets):
				high_pos -= 1
			while hit(mz,peaksets[high_pos].mz,mass_tol) and high_pos < len(peaksets):
				high_pos += 1	
				if high_pos >= len(peaksets):
					high_pos -= 1
					break	

			n_hits = 0
			best_hit = -1
			best_mass_error = 100
			best_ii = None
			for p in range(low_pos,high_pos+1):
				ii = IntensityInstance.objects.filter(peakset = peaksets[p],experiment = experiment)
				if len(ii) > 0:
					ii = ii[0]
					if abs(peaksets[p].rt - rt) < rt_tol:	
						n_hits += 1
						mass_error = abs((mz - peaksets[p].mz)/mz)
						if mass_error < best_mass_error:
							best_hit = p
							best_mass_error = mass_error
							best_ii = ii
				
			if n_hits > 0:
				best_ii.document = doc
				best_ii.save()
			
			n_done += 1
			if n_done % 100 == 0:
				print "\t\tDone {} documents".format(n_done)
		
		

	# raw_data = []
	# mz_list = []
	# rt_list = []
	# id_list = []
	# intensities = []
	# counts = []
	# count_hist = {}
	# with open(combined_file_name,'r') as f:
	# 	reader = csv.reader(f,delimiter='\t')
	# 	heads = reader.next()
	# 	for line in reader:
	# 		raw_data.append(line)
	# 		id = int(line[0])
	# 		mz = float(line[1])
	# 		rt = float(line[2])
	# 		new_intensity = [float(a) if not a=="" else 'nan' for a in line[3:]]
	# 		new_count = sum([1 for a in new_intensity if type(a) == float])
	# 		counts.append(new_count)
	# 		if new_count in count_hist:
	# 			count_hist[new_count] += 1
	# 		else:
	# 			count_hist[new_count] = 1
 # 			intensities.append(new_intensity)
 # 			mz_list.append(mz)
 # 			rt_list.append(rt)
 # 			id_list.append(id)

	# print "Loaded {} peaksets".format(len(raw_data))
	# print "Diagnostics: intensities present:"
	# ch = zip(count_hist.keys(),count_hist.values())
	# ch = sorted(ch,key = lambda x: x[0])

	# for c,h in ch:
	# 	print c,h


	# # Hack the names about to make them match - this should be provided to the script
	# # to avoid errors...fix!
	# experiment_match = {}
	# short_heads = []
	# for head in heads[3:]:
	# 	s = head.split('_')
	# 	new_name = s[0] + '_' + s[1] + '_' + s[2]
	# 	short_heads.append(new_name)
	# 	match = [e for e in experiments if new_name in e.name]
	# 	experiment_match[new_name] = match[0]

	

	# print "Making the peakset objects"
	# for i,mz_value in enumerate(mz_list):
	# 	rt_value = rt_list[i]
	# 	new_peakset = PeakSet.objects.get_or_create(multifileexperiment = mfe,mz = mz_value,rt = rt_value,
	# 												original_file = combined_file_name,original_id = id_list[i])[0]
	# 	for j,intense in enumerate(intensities[i]):
	# 		if not intense == 'nan':
	# 			exper = experiment_match[short_heads[j]]
	# 			ii = IntensityInstance.objects.get_or_create(peakset = new_peakset,experiment = exper)[0]
	# 			ii.intensity = intense
	# 			ii.save()
	# 	if i%100 == 0:
	# 		print "Done: {}".format(i)

