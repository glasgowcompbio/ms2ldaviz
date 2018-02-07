import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")



import django
django.setup()

from django.db import transaction

from basicviz.models import *

# Script to migrate the gnps_binned_005 experiment over to the binned_005 featureset

if __name__ == '__main__':
	e = Experiment.objects.get(id = 191)
	print e

	fs = BVFeatureSet.objects.get(name = 'binned_005')
	fs_features = Feature.objects.filter(featureset = fs)

	feat_dict = {}
	for f in fs_features:
		feat_dict[f.name] = f

	# gnps_docs = Document.objects.filter(experiment = e)

	# gnps_fi = FeatureInstance.objects.filter(document__in = gnps_docs)

	# gnps_fi = filter(lambda x: x.feature.featureset == None,gnps_fi)

	# total = len(gnps_fi)
	# n_done = 0

	# with transaction.atomic():
	# 	for fi in gnps_fi:
	# 		gnps_f = fi.feature
	# 		if not gnps_f.featureset:
	# 			# reformat name just in case
	# 			tokens = gnps_f.name.split('_')
	# 			newname = tokens[0] + "_{:.4f}".format(float(tokens[1]))
	# 			if newname in feat_dict:
	# 				global_feature = feat_dict[newname]
	# 				fi.feature = global_feature
	# 				fi.save()
	# 			else:
	# 				gnps_f.name = newname # ensure correct format
	# 				gnps_f.featureset = fs
	# 				gnps_f.experiment = None
	# 				gnps_f.save()
	# 		n_done += 1
	# 		if n_done % 1000 == 0:
	# 			print n_done,total

	# fix the features in the mass2motifs
	motifs = Mass2Motif.objects.filter(experiment = e)
	gnps_fmi = Mass2MotifInstance.objects.filter(mass2motif__in = motifs)
	total = len(gnps_fmi)
	n_done = 0
	print "MOTIFS"
	with transaction.atomic():
		for fmi in gnps_fmi:
			gnps_f = fmi.feature
			if not gnps_f.featureset:
				# reformat name just in case
				tokens = gnps_f.name.split('_')
				newname = tokens[0] + "_{:.4f}".format(float(tokens[1]))
				if newname in feat_dict:
					global_feature = feat_dict[newname]
					fmi.feature = global_feature
					fmi.save()
				else:
					print "feature in motif that wasnt in a document??"
			n_done += 1
			if n_done % 100 == 0:
				print n_done,total



	e.featureset = fs
	e.save()
