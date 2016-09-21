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

from basicviz.models import MultiFileExperiment,Experiment,Document

if __name__ == '__main__':
	multifile_experiment_name = sys.argv[1]
	mfe = MultiFileExperiment.objects.get(name = multifile_experiment_name)
	links = mfe.multilink_set.all()
	print "Found {} links".format(len(links))
	experiments = [l.experiment for l in links]
	for experiment in experiments:
		docs = experiment.document_set.all()
		print "Experiment {}, found {} documents".format(experiment,len(docs))
		for doc in docs:
			if '_' in doc.name:
				split_name = doc.name.split('_')
				mz = float(split_name[0])
				rt = float(split_name[1])
				md = jsonpickle.decode(doc.metadata)
				md['parentmass'] = mz
				md['mz'] = mz
				md['rt'] = rt
				doc.metadata = jsonpickle.encode(md)
				doc.save()
	# experiment = Experiment.objects.get(name = experiment_name)
	# documents = Document.objects.filter(experiment = experiment)
	# for document in documents:
	# 	md = jsonpickle.decode(document.metadata)
	# 	if 'm/z' in md:
	# 		md['parentmass'] = float(md['m/z'])
	# 	document.metadata = jsonpickle.encode(md)
	# 	document.save()