import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import numpy as np
import bisect
import jsonpickle

from basicviz.models import Experiment,Mass2Motif,Feature,Mass2MotifInstance,Alpha
from decomposition.models import FeatureSet,MotifSet,GlobalFeature,FeatureMap,GlobalMotif,GlobalMotifsToSets,Beta

# This script makes a motifset object and a beta object based on an Experiment

if __name__=='__main__':
    experiment = Experiment.objects.get(name = 'massbank_binned_005_alpha') 
    print("Got experiment ",experiment)

    featureset = FeatureSet.objects.get(name = 'binned_005')
    print("Got featureset ",featureset)

    experiment_motifs = Mass2Motif.objects.filter(experiment = experiment)
    print("Got {} motifs".format(len(experiment_motifs)))

    experiment_features = Feature.objects.filter(experiment = experiment)
    print("Got {} features".format(len(experiment_features)))


    n_added = 0
    n_done = 0
    for f in experiment_features:
        fe,status = GlobalFeature.objects.get_or_create(min_mz = f.min_mz,max_mz = f.max_mz,name = f.name,featureset = featureset)
        fmap,status2 = FeatureMap.objects.get_or_create(localfeature = f,globalfeature = fe)
        if status:
            n_added += 1
        n_done += 1
        if n_done % 1000 == 0:
            print(n_done,n_added)

    print("Added {} features to {}".format(n_added,featureset))

    motifset,status = MotifSet.objects.get_or_create(name = 'massbank_motifset_alpha',featureset = featureset)
    if status:
        print("Created {}".format(motifset))

    n_added = 0
    for motif in experiment_motifs:
        g,status = GlobalMotif.objects.get_or_create(originalmotif = motif)
        if status:
            n_added += 1
        gm,status = GlobalMotifsToSets.objects.get_or_create(motif = g,motifset = motifset)

    print("Added {} global motifs".format(n_added))

    gmms = GlobalMotifsToSets.objects.filter(motifset = motifset)
    global_motifs = [g.motif for g in gmms]
    print("{} motifs in motifset".format(len(global_motifs)))
    fmap = FeatureMap.objects.filter(localfeature__experiment = experiment)
    print("{} featuremap objects".format(len(fmap)))

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


    # beta = []
    # for m in range(n_motifs):
    #   beta.append([0 for f in range(n_features)])

    betalist = []

    originalmotifs = [m.originalmotif for m in global_motifs]
    fm2ms = Mass2MotifInstance.objects.filter(mass2motif__in = originalmotifs)
    print("Found {} instances".format(len(fm2ms)))
    n_done = 0
    for fm2m in fm2ms:
      n_done += 1
      if fm2m.feature in feature_map_dict:
          fpos = feature_index[feature_map_dict[fm2m.feature]]
          mpos = motif_index[motif_map_dict[fm2m.mass2motif]]
          betalist.append((mpos,fpos,fm2m.probability))
          # beta[mpos][fpos] = fm2m.probability
      if n_done % 100 == 0:
          print(n_done,len(fm2ms))

    
    # normalise
    # norm_beta = []
    # for row in beta:
    #     total = sum(row)
    #     if total > 0:
    #       row = [r/total for r in row]
    #     norm_beta.append(row)

    # beta = norm_beta


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

    # make a dummy experiment object
    # e = Experiment.objects.get_or_create(name = '')[0]
    e = experiment

    b,status = Beta.objects.get_or_create(experiment = experiment,motifset = motifset)

    b.beta = jsonpickle.encode(betalist)
    b.motif_id_list = jsonpickle.encode(motif_id_list)
    b.feature_id_list = jsonpickle.encode(feature_id_list)
    b.alpha_list = jsonpickle.encode(alpha_list)
    b.experiment = experiment
    b.motifset = motifset

    b.save()
