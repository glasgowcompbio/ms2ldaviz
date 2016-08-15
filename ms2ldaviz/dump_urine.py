import os
import csv
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")


import django

django.setup()

from basicviz.models import Experiment,Document,Mass2Motif,DocumentMass2Motif,FeatureInstance,FeatureMass2MotifInstance,Mass2MotifInstance

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    out_file = sys.argv[3]
    p_thresh = float(sys.argv[2])
    experiment = Experiment.objects.get(name = experiment_name)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    annotated_mass2motifs = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)

    documents = Document.objects.filter(experiment = experiment)
    annotated_documents = []
    for document in documents:
    	if document.annotation:
    		annotated_documents.append(document)

    print "Found {} m2ms and {} documents".format(len(annotated_mass2motifs),len(annotated_documents))

    for document in annotated_documents:
    	docm2ms = DocumentMass2Motif.objects.filter(document = document,
    												mass2motif__in = annotated_mass2motifs,
    												probability__get = p_thresh)
    	for docm2m in docm2ms:
	    	print "{}: {}, {}".format(document.annotation,docm2m.mass2motif.annotation,docm2m.probability)