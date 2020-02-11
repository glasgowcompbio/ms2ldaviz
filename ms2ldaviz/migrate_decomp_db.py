import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

from basicviz.models import Experiment,Document
from decomposition.models import FeatureSet,MotifSet,GlobalMotif,GlobalMotifsToSets,Decomposition,DocumentGlobalMass2Motif

if __name__ == '__main__':
	# find all decomposition experiments

	decomp_experiments = Experiment.objects.filter(experiment_type = "1")
	print("Decomposition Experiments:")
	for d in decomp_experiments:
		print("\t",d)
	print("")

	fs = FeatureSet.objects.get(name='binned_005')
	print("Feature Set:")
	print("\t",fs)
	print("")

	massbank_experiment = Experiment.objects.get(name = 'massbank_binned_005')
	print("Massbank Experiment:")
	print("\t",massbank_experiment)
	print("")

	ms,status = MotifSet.objects.get_or_create(name = 'massbank_motifset',featureset = fs)
	print("Motif Set")
	print("\t",ms,status)
	print("")


	motifs = GlobalMotif.objects.all()
	print("{} motifs in total".format(len(motifs)))
	motifs = GlobalMotif.objects.filter(originalmotif__experiment = massbank_experiment)
	print("{} motifs from {}".format(len(motifs),massbank_experiment))

	n_made = 0
	for motif in motifs:
		gm,status = GlobalMotifsToSets.objects.get_or_create(motifset = ms,motif = motif)
		if status:
			n_made += 1

	print("Made {} new motif to motifset connections".format(str(n_made)))

	print("")
	print("Making experiment <-> decomposition links")
	for d in decomp_experiments:
		# Make a decomposition object
		de,status = Decomposition.objects.get_or_create(name = d.name + " decomposition",experiment = d,motifset = ms)
		print("\t",de,status)
		
		docs = Document.objects.filter(experiment = d)
		dm2ms = DocumentGlobalMass2Motif.objects.filter(document__in = docs)
		print("\tFound {} dm2ms, adding decomposition to links".format(len(dm2ms)))
		for dm2m in dm2ms:
			dm2m.decomposition = de
			dm2m.save()

	
