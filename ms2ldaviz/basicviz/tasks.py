import bisect
import numpy as np

from ms2ldaviz.celery_tasks import app
from basicviz.models import Experiment,Mass2MotifInstance,MotifMatch

@app.task
def match_motifs(experiment_id,base_experiment_id):
    
    min_score_to_save = 0.5

    experiment = Experiment.objects.get(id = experiment_id)
    base_experiment = Experiment.objects.get(id = base_experiment_id)


    # Get the features associated with this experiment for matching
    features = experiment.feature_set.all().order_by('min_mz')
    base_features = base_experiment.feature_set.all().order_by('min_mz')

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
    base_mz = [float(f.name.split('_')[1]) for f in base_fragments]
    for i,f in enumerate(base_fragments):
        if not f.min_mz:
            f.min_mz = base_mz[i] - 5e-6 * base_mz[i]
        if not f.max_mz:
            f.max_mz = base_mz[i] + 5e-6 * base_mz[i]

    base_mz = [float(f.name.split('_')[1]) for f in base_losses]
    for i,f in enumerate(base_losses):
        if not f.min_mz:
            f.min_mz = base_mz[i] - 5e-6 * base_mz[i]
        if not f.max_mz:
            f.max_mz = base_mz[i] + 5e-6 * base_mz[i]

    base_fragments = sorted(base_fragments,key = lambda x: x.min_mz)
    base_losses = sorted(base_losses,key = lambda x: x.min_mz)

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
