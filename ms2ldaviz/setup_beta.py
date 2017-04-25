import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import numpy as np
import bisect
import jsonpickle
from decomposition.models import FeatureSet,GlobalFeature,Decomposition,FeatureMap,GlobalMotif,Beta,DocumentGlobalFeature
from basicviz.models import Experiment,Feature,Mass2Motif,Mass2MotifInstance,Alpha,Document

if __name__ == '__main__':
    # Match all of the features in massbank to this experiment
    massbank_experiment = Experiment.objects.get(name = 'massbank_binned_005_alpha')
    massbank_features = Feature.objects.filter(experiment = massbank_experiment)
    fs = FeatureSet.objects.get_or_create(name='binned_005')[0]

    n_done = 0
    for localfeature in massbank_features:
      gf = GlobalFeature.objects.get_or_create(name = localfeature.name,
                                          min_mz = localfeature.min_mz,
                                          max_mz = localfeature.max_mz,
                                          featureset = fs)[0]
      FeatureMap.objects.get_or_create(localfeature = localfeature,
                                      globalfeature = gf)
      n_done += 1
      if n_done % 100 == 0:
          print n_done,len(massbank_features)

    # Create the motif links
    massbank_motifs = Mass2Motif.objects.filter(experiment = massbank_experiment)
    for motif in massbank_motifs:
      GlobalMotif.objects.get_or_create(originalmotif = motif)

    massbank_global_motifs = GlobalMotif.objects.filter(originalmotif__experiment = massbank_experiment).order_by('originalmotif__name')
    n_motifs = len(massbank_global_motifs)
    massbank_features = FeatureMap.objects.filter(localfeature__experiment = massbank_experiment).order_by('globalfeature__name')
    n_features = len(massbank_features)

    feature_map_dict = {}
    for feature in massbank_features:
      feature_map_dict[feature.localfeature] = feature.globalfeature

    motif_map_dict = {}
    for globalmotif in massbank_global_motifs:
      motif_map_dict[globalmotif.originalmotif] = globalmotif

    motif_index = {}
    for i in range(n_motifs):
      motif_index[massbank_global_motifs[i]] = i
    feature_index = {}
    for i in range(n_features):
      feature_index[massbank_features[i].globalfeature] = i


    beta = []
    for m in range(n_motifs):
      beta.append([0 for f in range(n_features)])

    originalmotifs = [m.originalmotif for m in massbank_global_motifs]
    fm2ms = Mass2MotifInstance.objects.filter(mass2motif__in = originalmotifs)
    print "Found {} instances".format(len(fm2ms))
    n_done = 0
    for fm2m in fm2ms:
      n_done += 1
      if fm2m.feature in feature_map_dict:
          fpos = feature_index[feature_map_dict[fm2m.feature]]
          mpos = motif_index[motif_map_dict[fm2m.mass2motif]]
          beta[mpos][fpos] = fm2m.probability
      if n_done % 100 == 0:
          print n_done,len(fm2ms)
    print "created beta"
    print np.array(beta).sum()

    b = Beta.objects.get_or_create(experiment = massbank_experiment)[0]

    # normalise
    norm_beta = []
    for row in beta:
      total = sum(row)
      if total > 0:
          row = [r/total for r in row]
      norm_beta.append(row)

    beta = norm_beta
    feature_id_list = [None for f in range(n_features)]
    motif_id_list = [None for m in range(n_motifs)]
    for motif,pos in motif_index.items():
      motif_id_list[pos] = motif.id
    for feature,pos in feature_index.items():
      feature_id_list[pos] = feature.id

    b.beta = jsonpickle.encode(beta)
    b.motif_id_list = jsonpickle.encode(motif_id_list)
    b.feature_id_list = jsonpickle.encode(feature_id_list)

    # Get the alphas
    alpha_list = [0.0 for m in range(n_motifs)]
    for motif,pos in motif_index.items():
      originalmotif = motif.originalmotif
      alpha = Alpha.objects.get(mass2motif = originalmotif)
      alpha_list[pos] = alpha.value

    b.alpha_list = jsonpickle.encode(alpha_list)

    b.save()
