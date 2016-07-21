import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Document,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,DocumentMass2Motif,FeatureMass2MotifInstance

if __name__ == '__main__':
<<<<<<< HEAD
	filename = '/home/beer3.dict'
=======
	filename = sys.argv[1]
	# filename = '/Users/simon/git/lda/notebooks/beer3.dict'
>>>>>>> 9c8207ec007a6b79962ca191368649000d946b0e
	with open(filename,'r') as f:
		lda_dict = pickle.load(f)
		experiment_name = filename.split('/')[-1].split('.')[0]
	experiment = Experiment.objects.get_or_create(name=experiment_name)[0]
	print "Loading corpus"
	n_done = 0
	to_do = len(lda_dict['corpus'])
	for doc in lda_dict['corpus']:
		n_done += 1
		if n_done % 100 == 0:
			print "Done {}/{}".format(n_done,to_do)
		metdat = jsonpickle.encode(lda_dict['doc_metadata'][doc])
		d = Document.objects.get_or_create(name=doc,experiment=experiment,metadata=metdat)[0]
		for word in lda_dict['corpus'][doc]:
			feature = Feature.objects.get_or_create(name=word,experiment=experiment)[0]
			fi = FeatureInstance.objects.get_or_create(document = d,feature = feature, intensity = lda_dict['corpus'][doc][word])
	print "Loading topics"
	n_done = 0
	to_do = len(lda_dict['beta'])
	for topic in lda_dict['beta']:
		n_done += 1
		if n_done % 100 == 0:
			print "Done {}/{}".format(n_done,to_do)
		metadata = {}
		metadata = lda_dict['topic_metadata'].get(topic,{})
		m2m = Mass2Motif.objects.get_or_create(name = topic,experiment = experiment,metadata = jsonpickle.encode(metadata))[0]
		for word in lda_dict['beta'][topic]:
			feature = Feature.objects.get(name = word,experiment=experiment)
			Mass2MotifInstance.objects.get_or_create(feature = feature,mass2motif = m2m,probability = lda_dict['beta'][topic][word])
	print "Loading theta"
	n_done = 0
	to_do = len(lda_dict['theta'])
	for doc in lda_dict['theta']:
		n_done += 1
		if n_done % 100 == 0:
			print "Done {}/{}".format(n_done,to_do)		
		document = Document.objects.get(name = doc,experiment=experiment)
		for topic in lda_dict['theta'][doc]:
			mass2motif = Mass2Motif.objects.get(name = topic,experiment = experiment)
			DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc][topic])
	print "Loading phi"
	n_done = 0
	to_do = len(lda_dict['phi'])
	for doc in lda_dict['phi']:
		n_done += 1
		if n_done % 100 == 0:
			print "Done {}/{}".format(n_done,to_do)
		document = Document.objects.get(name = doc,experiment=experiment)
		for word in lda_dict['phi'][doc]:
			feature = Feature.objects.get(name=word,experiment=experiment)
			feature_instance = FeatureInstance.objects.get(document=document,feature=feature)
			for topic in lda_dict['phi'][doc][word]:
				mass2motif = Mass2Motif.objects.get(name=topic,experiment=experiment)
				probability = lda_dict['phi'][doc][word][topic]
				FeatureMass2MotifInstance.objects.get_or_create(featureinstance = feature_instance,mass2motif = mass2motif,probability = probability)
