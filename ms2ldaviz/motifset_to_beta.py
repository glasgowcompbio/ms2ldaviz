# A script to turn a motifset object into a beta object
import os
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from decomposition.models import MotifSet,FeatureSet,GlobalFeature,GlobalMotifsToSets,Beta,FeatureMap
from basicviz.models import Alpha,Mass2MotifInstance

if __name__ == '__main__':
    motifset_name = sys.argv[1]
    ms = MotifSet.objects.get(name = motifset_name)
    print "Loaded {}".format(ms)

    fs = FeatureSet.objects.get(motifset = ms)
    print "Extracted {}".format(fs)

    motif_links = GlobalMotifsToSets.objects.filter(motifset = ms)
    global_motifs = [m.motif for m in motif_links]

    global_features = GlobalFeature.objects.filter(featureset = fs)
    print "Extracted {} global features".format(len(global_features))
    fmap = FeatureMap.objects.filter(globalfeature__in = global_features)
    print "Extracted {} feature map objects".format(len(fmap))


    feature_map_dict = {}
    for feature in fmap:
      feature_map_dict[feature.localfeature] = feature.globalfeature

    motif_map_dict = {}
    for globalmotif in global_motifs:
      motif_map_dict[globalmotif.originalmotif] = globalmotif

    n_motifs = len(global_motifs)
    n_features = len(fmap)

    motif_index = {}
    for i in range(n_motifs):
        motif_index[global_motifs[i]] = i
    feature_index = {}
    for i in range(n_features):
        feature_index[fmap[i].globalfeature] = i

        betalist = []

    originalmotifs = [m.originalmotif for m in global_motifs]
    fm2ms = Mass2MotifInstance.objects.filter(mass2motif__in = originalmotifs)
    print "Found {} instances".format(len(fm2ms))
    n_done = 0
    for fm2m in fm2ms:
      n_done += 1
      if fm2m.feature in feature_map_dict:
          fpos = feature_index[feature_map_dict[fm2m.feature]]
          mpos = motif_index[motif_map_dict[fm2m.mass2motif]]
          betalist.append((mpos,fpos,fm2m.probability))
          # beta[mpos][fpos] = fm2m.probability
      if n_done % 100 == 0:
          print n_done,len(fm2ms)

    feature_id_list = [None for f in range(n_features)]
    motif_id_list = [None for m in range(n_motifs)]
    for motif,pos in motif_index.items():
      motif_id_list[pos] = motif.id
    for feature,pos in feature_index.items():
      feature_id_list[pos] = feature.id

    # Get the alphas
    alpha_list = [0.0 for m in range(n_motifs)]
    for motif,pos in motif_index.items():
        originalmotif = motif.originalmotif
        alpha = Alpha.objects.get(mass2motif = originalmotif)
        alpha_list[pos] = alpha.value


    b,status = Beta.objects.get_or_create(motifset = motifset)

    b.beta = jsonpickle.encode(betalist)
    b.motif_id_list = jsonpickle.encode(motif_id_list)
    b.feature_id_list = jsonpickle.encode(feature_id_list)
    b.alpha_list = jsonpickle.encode(alpha_list)
    b.motifset = motifset

    b.save()

