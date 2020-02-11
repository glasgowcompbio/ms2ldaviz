from ms2ldaviz.celery_tasks import app

import numpy as np

from basicviz.models import Experiment,Mass2Motif,Mass2MotifInstance,MotifMatch
from motifdb.models import MDBMotifSet

from basicviz.tasks import get_experiment_features,map_the_features

@app.task
def start_motif_matching_task(experiment_id,motif_set_id,min_score_to_save):
    print(experiment_id,motif_set_id)
    motif_set = MDBMotifSet.objects.get(id = motif_set_id)
    experiment = Experiment.objects.get(id = experiment_id)

    # get the fs objects
    fs = experiment.featureset
    base_fs = motif_set.featureset

    print(fs,base_fs)

    # get the features used in each experiment
    features = get_experiment_features(experiment)
    base_features = get_motifset_features(motif_set)

    
    if not fs == None and not base_fs == None:
        feature_map = map_the_features(fs,base_fs,features,base_features)
    else:
        # TODO: put in a useful response here!
        return None

    print("Found matches of {} out of {} features".format(len(feature_map),len(features)))

    # get the motifs
    motifs = experiment.mass2motif_set.all()
    motif_dict = {}
    motif_norms = {}

    base_motifs = motif_set.mdbmotif_set.all()
    base_motif_dict = {}
    base_motif_norms = {}


    # get the instances and compute the normalisation terms
    mfs = Mass2MotifInstance.objects.filter(mass2motif__in = motifs)
    base_mfs = Mass2MotifInstance.objects.filter(mass2motif__in = base_motifs)

    # Store them in dicts for quick access
    for mf in mfs:
        motif = mf.mass2motif
        if not motif in motif_dict:
            motif_dict[motif] = {}
            motif_norms[motif] = 0.0
        motif_dict[motif][mf.feature] = mf.probability
        motif_norms[motif] += mf.probability ** 2

    for mf in base_mfs:
        motif = mf.mass2motif
        if not motif in base_motif_dict:
            base_motif_dict[motif] = {}
            base_motif_norms[motif] = 0.0
        base_motif_dict[motif][mf.feature] = mf.probability
        base_motif_norms[motif] += mf.probability ** 2


    for motif in motif_norms:
        motif_norms[motif] = np.sqrt(motif_norms[motif])
    for motif in base_motif_norms:
        base_motif_norms[motif] = np.sqrt(base_motif_norms[motif])


    # compute the cosine scores
    matches = []
    for motif in motif_dict.keys():
        best_score = 0.0
        best_base = None
        best_list = None
        for base_motif in base_motif_dict.keys():
            match_list = []
            score = 0.0
            # Note: with matching now, one feature could hit two base features. In this case
            # we can't use it twice or the normalisation will mess up. So, we use the assignment
            # that gives the max score
            for feature,probability in motif_dict[motif].items():
                temp_list = None
                if feature in feature_map:
                    map_features = feature_map[feature]
                    for map_feature in map_features:
                        temp_score = 0.0
                        if map_feature in base_motif_dict[base_motif]:
                            map_probability = base_motif_dict[base_motif][map_feature]
                            t = probability * map_probability
                            if t >= temp_score:
                                temp_score = probability * map_probability
                                temp_list = (feature,probability,map_feature,map_probability)
                    match_list.append(temp_list)
                    score += temp_score
            
            score /= motif_norms[motif]
            score /= base_motif_norms[base_motif]
            
            
            if score > best_score: # only keep the best one from this query motif
                best_score = score
                best_base = base_motif
                best_list = match_list
        if best_score >= min_score_to_save:
            matches.append((motif,best_base,best_score,best_list))
            print(motif,best_score,best_base)

    for match in matches:
        frommotif = match[0]
        tomotif = match[1]
        score = match[2]
        mm,status = MotifMatch.objects.get_or_create(frommotif = frommotif,tomotif = tomotif)
        mm.score = score
        mm.save()


def get_motifset_features(motif_set):
    motifs = motif_set.mdbmotif_set.all()
    feature_instances = Mass2MotifInstance.objects.filter(mass2motif__in = motifs)
    features = set([fi.feature for fi in feature_instances])
    return features