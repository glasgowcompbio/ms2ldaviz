import os
import pickle
import numpy as np
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment,Document,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,DocumentMass2Motif,FeatureMass2MotifInstance

if __name__ == '__main__':
	filename = '/Users/simon/git/lda/notebooks/beer3.dict'
	with open(filename,'r') as f:
		lda_dict = pickle.load(f)
	experiment = Experiment.objects.get_or_create(name='beer3')[0]

	# for doc in lda_dict['corpus']:
	# 	print doc
	# 	metdat = jsonpickle.encode(lda_dict['doc_metadata'][doc])
	# 	d = Document.objects.get_or_create(name=doc,experiment=experiment,metadata=metdat)[0]
	# 	for word in lda_dict['corpus'][doc]:
	# 		feature = Feature.objects.get_or_create(name=word,experiment=experiment)[0]
	# 		fi = FeatureInstance.objects.get_or_create(document = d,feature = feature, intensity = lda_dict['corpus'][doc][word])

	# for topic in lda_dict['beta']:
	# 	m2m = Mass2Motif.objects.get_or_create(name = topic,experiment = experiment)[0]
	# 	for word in lda_dict['beta'][topic]:
	# 		feature = Feature.objects.get(name = word,experiment=experiment)
	# 		Mass2MotifInstance.objects.get_or_create(feature = feature,mass2motif = m2m,probability = lda_dict['beta'][topic][word])

	# for doc in lda_dict['theta']:
	# 	document = Document.objects.get(name = doc,experiment=experiment)
	# 	print document
	# 	for topic in lda_dict['theta'][doc]:
	# 		mass2motif = Mass2Motif.objects.get(name = topic,experiment = experiment)
	# 		DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc][topic])

	for doc in lda_dict['phi']:
		document = Document.objects.get(name = doc,experiment=experiment)
		print document
		for word in lda_dict['phi'][doc]:
			feature = Feature.objects.get(name=word,experiment=experiment)
			feature_instance = FeatureInstance.objects.get(document=document,feature=feature)
			for topic in lda_dict['phi'][doc][word]:
				mass2motif = Mass2Motif.objects.get(name=topic,experiment=experiment)
				probability = lda_dict['phi'][doc][word][topic]
				FeatureMass2MotifInstance.objects.get_or_create(featureinstance = feature_instance,mass2motif = mass2motif,probability = probability)