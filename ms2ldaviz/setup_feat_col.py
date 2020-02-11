import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

from basicviz.models import *

if __name__ == '__main__':


	other_experiment_name = sys.argv[1]

	fs,status = BVFeatureSet.objects.get_or_create(name = 'binned_005')
	if status:
		print("Created feature set")
	else:
		print("Featureset already exists")

	mbe = Experiment.objects.get(name = 'massbank_binned_005')
	print("Got " + str(mbe))
	if mbe.featureset == None:
		mbe.featureset = fs
		mbe.save()

	mbe_features = Feature.objects.filter(experiment = mbe)
	print("Got {} features".format(len(mbe_features)))

	mbe_features_sub = Feature.objects.filter(experiment = mbe,featureset = None)
	print("{} have no featureset".format(len(mbe_features_sub)))
	for f in mbe_features_sub:
		f.featureset = fs
		f.save()


	# Now get the features as tied to the feature set
	mbe_features = Feature.objects.filter(featureset = fs)
	print("Got {} features".format(len(mbe_features)))

	fnames = set([f.name for f in mbe_features])

	# get another experiment
	i93 = Experiment.objects.get(name = other_experiment_name)
	i93.featureset = fs
	i93.save()
	print("Got " + str(i93))

	i93_features = Feature.objects.filter(experiment = i93)

	print("Got {} features".format(len(i93_features)))
	for f in i93_features:
		if f.name in fnames:
			# Find all the instances
			fis = FeatureInstance.objects.filter(feature = f)
			gfeature = [g for g in mbe_features if g.name == f.name][0]
			for fi in fis:
				fi.feature = gfeature
				fi.save()
			mis = Mass2MotifInstance.objects.filter(feature = f)
			for ms in mis:
				ms.feature = gfeature
				ms.save()


		else:
			new_feature = Feature.objects.create(name = f.name,featureset = fs,min_mz = f.min_mz,max_mz = f.max_mz)
			fis = FeatureInstance.objects.filter(feature = f)
			for fi in fis:
				fi.feature = new_feature
				fi.save()
			mis = Mass2MotifInstance.objects.filter(feature = f)
			for ms in mis:
				ms.feature = new_feature
				ms.save()


	for f in i93_features:
		if len(f.featureinstance_set.all()) == 0 and len(f.mass2motifinstance_set.all()) == 0 and len(f.featuremap_set.all()) == 0:
			f.delete()
		else:
			print(f)