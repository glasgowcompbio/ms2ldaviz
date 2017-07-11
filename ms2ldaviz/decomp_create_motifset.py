import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from decomposition.models import MotifSet,FeatureSet,GlobalFeature,GlobalMotifsToSets,Beta,FeatureMap
from basicviz.models import Alpha,Mass2MotifInstance,Experiment,Document,Mass2Motif,FeatureInstance


# Script to transform an experiment into a motifset for decomposition
if __name__ == '__main__':
    experiment_name = sys.argv[1]
    original_experiment = Experiment.objects.get(name = experiment_name)
    docs = Document.objects.filter(experiment = original_experiment)
    

    # Find out the featureset
    temp_doc = docs[0]
    fi = FeatureInstance.objects.filter(document = temp_doc)[0]
    bfs = fi.feature.featureset

    original_features = Feature.objects.filter(featureset = bfs)
    

    # Get the decomposition featureset - hardcoded
    fs = FeatureSet.objects.get_or_create(name='binned_005')[0]

    n_done = 0
    for localfeature in original_features:
      gf = GlobalFeature.objects.get_or_create(name = localfeature.name,
                                          min_mz = localfeature.min_mz,
                                          max_mz = localfeature.max_mz,
                                          featureset = fs)[0]
      FeatureMap.objects.get_or_create(localfeature = localfeature,
                                      globalfeature = gf)
      n_done += 1
      if n_done % 100 == 0:
          print n_done,len(original_features)

    # Create the motif links
    global_motifs = []
    original_motifs = Mass2Motif.objects.filter(experiment = original_experiment)
    for motif in massbank_motifs:
      gm = GlobalMotif.objects.get_or_create(originalmotif = motif)[0]
      global_motifs.append(gm)

    # Create the motifset and put the global motifs in it
    ms = MotifSet.objects.get_or_create(name = sys.argv[2],featureset = fs)[0]
    for gm in global_motifs:
      gms = GlobalMotifsToSets.objects.get_or_create(motif = gm,motifset = ms)

