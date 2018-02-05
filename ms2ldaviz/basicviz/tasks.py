import bisect
import numpy as np

from ms2ldaviz.celery_tasks import app
from basicviz.models import Experiment,Mass2MotifInstance,MotifMatch


def get_experiment_features(experiment):
    motifs = experiment.mass2motif_set.all()
    feature_instances = Mass2MotifInstance.objects.filter(mass2motif__in = motifs)
    features = set([fi.feature for fi in feature_instances])
    return features

@app.task
def match_motifs_set(experiment_id,base_experiment_id,min_score_to_save = 0.5):

    # Get the expeiment objetcs 
    experiment = Experiment.objects.get(id = experiment_id)
    base_experiment = Experiment.objects.get(id = base_experiment_id)

    # get the fs objects
    fs = experiment.featureset
    base_fs = base_experiment.featureset

    print fs,base_fs

    # get the features used in each experiment
    features = get_experiment_features(experiment)
    base_features = get_experiment_features(base_experiment)

    base_feature_name_dict = {}
    for feature in base_features:
        base_feature_name_dict[feature.name] = feature


    feature_map = {}
    if fs == base_fs:
        # the task is easy because both sets of motifs are in the same set
        for feature in features:
            if feature in base_features:
                feature_map[feature] = [feature]
    elif fs.name == 'binned_01' and base_fs.name == 'binned_005':
        # allow the comparison of things with wider fs with the original motifs
        for feature in features:
            ftype = feature.name.split('_')[0]
            # work out the names of the two 005 features
            lowval = feature.min_mz
            highval = feature.max_mz
            fname1 = ftype + "_{:.4f}".format(lowval + 0.005/2.0)
            fname2 = ftype + "_{:.4f}".format(highval - 0.005/2.0)
            temp = []
            if fname1 in base_feature_name_dict:
                temp.append(base_feature_name_dict[fname1])
            if fname2 in base_feature_name_dict:
                temp.append(base_feature_name_dict[fname2])
            if len(temp) > 0:
                feature_map[feature] = temp
    elif fs.name == 'binned_005' and basefs.name == 'binned_01':
        # Each will only match to one, but the matches will appear more than once
        for feature in features:
            ftype = feature.name.split('_')[0]
            lowval = feature.min_mz
            highval = feature.max_mz
            # Each feature will map perfectly into the first or second half of 
            # one of the base features
            # we just need to work out if it is a first half or a second half one
            # Easiest way is to check if upper lor lower val is a name in the other set
            uppername = ftype+ "_{:.4f}".format(highval)
            if uppername in base_feature_name_dict:
                feature_map[feature] = base_feature_name_dict[uppername]
            else:
                lowername = ftype+ "_{:.4f}".format(lowval)
                if lowername in base_feature_name_dict:
                    feature_map[lowername] = base_feature_name_dict[lowername]
    else:
        # Ensure backward compatibility
        return match_motifs(experiment_id,base_experiment_id,min_score_to_save = min_score_to_save)

    mzd_features = filter(lambda x: x.name.startswith('mzdiff'),feature_map.keys())
    for f in mzd_features:
        print f,feature_map[f]


    print "Found matches of {} out of {} features".format(len(feature_map),len(features))

    # get the motifs
    motifs = experiment.mass2motif_set.all()
    motif_dict = {}
    motif_norms = {}

    base_motifs = base_experiment.mass2motif_set.all()
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
            for feature,probability in motif_dict[motif].items():
                if feature in feature_map:
                    map_features = feature_map[feature]
                    for map_feature in map_features:
                        if map_feature in base_motif_dict[base_motif]:
                            map_probability = base_motif_dict[base_motif][map_feature]
                            score += probability * map_probability
                            match_list.append((feature,probability,map_feature,map_probability))
            
            score /= motif_norms[motif]
            score /= base_motif_norms[base_motif]
            
            
            if score > best_score:
                best_score = score
                best_base = base_motif
                best_list = match_list
        if best_score >= min_score_to_save:
            matches.append((motif,best_base,best_score,best_list))
            print motif,best_score,best_base

    for match in matches:
        frommotif = match[0]
        tomotif = match[1]
        score = match[2]
        mm,status = MotifMatch.objects.get_or_create(frommotif = frommotif,tomotif = tomotif)
        mm.score = score
        mm.save()




@app.task
def match_motifs(experiment_id,base_experiment_id,min_score_to_save = 0.5):
    

    experiment = Experiment.objects.get(id = experiment_id)
    base_experiment = Experiment.objects.get(id = base_experiment_id)


    # Get the features associated with this experiment for matching
    # features = experiment.feature_set.all().order_by('min_mz')
    # base_features = base_experiment.feature_set.all().order_by('min_mz')
    features = get_experiment_features(experiment)
    base_features = get_experiment_features(base_experiment)
    features = sorted(features,key = lambda x: x.min_mz)
    base_features = sorted(base_features,key = lambda x: x.min_mz)

    print len(features)
    print len(base_features)

    feature_name_dict = {}
    base_feature_name_dict = {}
    for feature in features:
        feature_name_dict[feature.name] = feature
    for feature in base_features:
        base_feature_name_dict[feature.name] = feature

    # Split into features and losses in case we need to match by mz
    base_fragments = [f for f in base_features if f.name.startswith('fragment')]
    base_losses = [f for f in base_features if f.name.startswith('loss')]


    # The following code is just for testing should be removed...

    # base_mz = [float(f.name.split('_')[1]) for f in base_fragments]
    # for i,f in enumerate(base_fragments):
    #     if not f.min_mz:
    #         f.min_mz = base_mz[i] - 5e-6 * base_mz[i]
    #     if not f.max_mz:
    #         f.max_mz = base_mz[i] + 5e-6 * base_mz[i]

    # base_mz = [float(f.name.split('_')[1]) for f in base_losses]
    # for i,f in enumerate(base_losses):
    #     if not f.min_mz:
    #         f.min_mz = base_mz[i] - 5e-6 * base_mz[i]
    #     if not f.max_mz:
    #         f.max_mz = base_mz[i] + 5e-6 * base_mz[i]

    # base_fragments = sorted(base_fragments,key = lambda x: x.min_mz)
    # base_losses = sorted(base_losses,key = lambda x: x.min_mz)

    # END OF TESTING BLOCK



    base_fragment_min_mz = [f.min_mz for f in base_fragments]
    base_fragment_max_mz = [f.max_mz for f in base_fragments]

    base_loss_min_mz = [f.min_mz for f in base_losses]
    base_loss_max_mz = [f.max_mz for f in base_losses]

    feature_map = {}

    # Here we map the features across the experiments.
    # The code first tries to match based on the name 
    # and then reverts to finding the lightest feature 
    # in the base set that includes the feature from the 
    # original set.

    for feature_name in feature_name_dict:
        feature = feature_name_dict[feature_name]
        if feature_name in base_feature_name_dict:
            feature_map[feature] = base_feature_name_dict[feature_name]
        else:
            feature_type = feature_name.split('_')[0]
            feature_mz = float(feature_name.split('_')[1])
            if feature_type == 'fragment':
                if feature_mz >= base_fragment_min_mz[0] and feature_mz <= base_fragment_max_mz[-1]:
                    pos = bisect.bisect_right(base_fragment_min_mz,feature_mz) - 1
                    if feature_mz <= base_fragment_max_mz[pos]:
                        feature_map[feature] = base_fragments[pos]
            elif feature_type == 'loss':
                if feature_mz >= base_loss_min_mz[0] and feature_mz <= base_loss_max_mz[-1]:
                    pos = bisect.bisect_right(base_loss_min_mz,feature_mz) - 1
                    if feature_mz <= base_loss_max_mz[pos]:
                        feature_map[feature] = base_losses[pos]



    # Extract the motifs from each experiment
    motifs = experiment.mass2motif_set.all()
    motif_dict = {}
    motif_norms = {}

    base_motifs = base_experiment.mass2motif_set.all()
    base_motif_dict = {}
    base_motif_norms = {}


    
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

    # Compute the cosine scores between all pairs
    matches = []
    for motif in motif_dict.keys():
        best_score = 0.0
        best_base = None
        best_list = None
        for base_motif in base_motif_dict.keys():
            match_list = []
            score = 0.0
            for feature,probability in motif_dict[motif].items():
                if feature in feature_map:
                    map_feature = feature_map[feature]
                    if map_feature in base_motif_dict[base_motif]:
                        map_probability = base_motif_dict[base_motif][map_feature]
                        score += probability * map_probability
                        match_list.append((feature,probability,map_feature,map_probability))
            
            score /= motif_norms[motif]
            score /= base_motif_norms[base_motif]
            
            
            if score > best_score:
                best_score = score
                best_base = base_motif
                best_list = match_list
        if best_score >= min_score_to_save:
            matches.append((motif,best_base,best_score,best_list))
            print motif,best_score,best_base

    for match in matches:
        frommotif = match[0]
        tomotif = match[1]
        score = match[2]
        mm,status = MotifMatch.objects.get_or_create(frommotif = frommotif,tomotif = tomotif)
        mm.score = score
        mm.save()
