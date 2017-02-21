import numpy as np
import bisect
import jsonpickle
from scipy.special import psi as psi

from decomposition.models import DocumentGlobalFeature,GlobalFeature,GlobalMotif,DocumentGlobalMass2Motif,DocumentFeatureMass2Motif
from basicviz.models import VizOptions


def decompose(documents,betaobject,normalise = 1000.0,store_threshold = 0.01):
    # Load the beta objects
    print "Loading and unpickling beta"
    beta = jsonpickle.decode(betaobject.beta)
    alpha = jsonpickle.decode(betaobject.alpha_list)
    motif_id_list = jsonpickle.decode(betaobject.motif_id_list)
    feature_id_list = jsonpickle.decode(betaobject.feature_id_list)

    alpha_matrix = np.array(alpha)
    beta_matrix = np.array(beta)

    word_index = {}
    for i,word_id in enumerate(feature_id_list):
        feature = GlobalFeature.objects.get(id = word_id)
        word_index[feature] = i

    motif_index = {}
    motif_list = []
    for i,motif_id in enumerate(motif_id_list):
        motif = GlobalMotif.objects.get(id = motif_id)
        motif_index[motif] = i
        motif_list.append(motif)

    K = len(motif_list)
    print "Performing e-steps"
    for document in documents:
        print document.name
        docfeatures = DocumentGlobalFeature.objects.filter(document = document)

        doc_dict = {}
        maxi = 0.0
        for df in docfeatures:
            doc_dict[df.feature] = df.intensity
            if df.intensity > maxi:
                maxi = df.intensity

        # normalise
        if normalise:
            for word in doc_dict:
                doc_dict[word] = int(normalise*doc_dict[word]/maxi)

        # Do the e-steps for this document
        phi_matrix = {}
        for word in doc_dict:
            phi_matrix[word] = None
        gamma = np.ones(K)
        for i in range(100): # do 100 iterations
            # print "Iteration {}".format(i)
            temp_gamma = np.zeros(K) + alpha_matrix
            for word,intensity in doc_dict.items():
                # Find the word position in beta
                word_pos = word_index[word]
                if beta_matrix[:,word_pos].sum() > 0:
                    log_phi_matrix = np.log(beta_matrix[:,word_pos]) + psi(gamma).T
                    log_phi_matrix = np.exp(log_phi_matrix - log_phi_matrix.max())
                    phi_matrix[word] = log_phi_matrix/log_phi_matrix.sum()
                    temp_gamma += phi_matrix[word]*intensity

            gamma = temp_gamma.copy()

        
        
        # normalise the gamma to get probabilities
        theta = gamma/gamma.sum()
        theta = list(theta)

        theta_motif = zip(theta,motif_list)
        theta_motif = sorted(theta_motif,key = lambda x : x[0],reverse = True)
        theta,motif = zip(*theta_motif)
        tot_prob = 0.0
        for i in range(K):
            if theta[i] < store_threshold:
                break
            motif_pos = motif_index[motif[i]]
            overlap_score = compute_overlap(phi_matrix,motif_pos,beta_matrix[motif_pos,:],word_index)
            print theta[i],overlap_score,motif[i].originalmotif.name,motif[i].originalmotif.annotation
            dgm2m,status = DocumentGlobalMass2Motif.objects.get_or_create(document = document,mass2motif = motif[i])
            dgm2m.probability = theta[i]
            dgm2m.overlap_score = overlap_score
            dgm2m.save()
            for word in phi_matrix:
                if phi_matrix[word] == None:
                    continue
                else:
                    if phi_matrix[word][motif_pos] >= store_threshold:
                        dfm = DocumentFeatureMass2Motif.objects.get_or_create(docm2m = dgm2m,feature = word)[0]
                        dfm.probability = phi_matrix[word][motif_pos]
                        dfm.save()





def compute_overlap(phi_matrix,motif_pos,beta_row,word_index):
    overlap_score = 0.0
    for word in phi_matrix:
        word_pos = word_index[word]
        if phi_matrix[word] == None:
            continue
        else:
            overlap_score += phi_matrix[word][motif_pos]*beta_row[word_pos]
    return overlap_score

def get_parents_decomposition(motif_id,vo_id):
    viz_options = VizOptions.objects.get(id = vo_id)
    motif = GlobalMotif.objects.get(id = motif_id)
    parent_data = []
    edge_choice = viz_options.edge_choice
    if edge_choice == 'probability':
        docm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = viz_options.edge_thresh).order_by('-probability')
    else:
        docm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = viz_options.edge_thresh).order_by('-overlap_score')
    for dm in docm2m:
        document = dm.document
        parent_data.append(get_parent_for_plot_decomp(document,motif = motif,edge_choice = edge_choice))

    return parent_data

def get_parent_for_plot_decomp(document,motif = None,score_type = 'probability',get_key = False):
    plot_data = []
    colours = ['red', 'green', 'black', 'yellow']
    docm2m = DocumentGlobalMass2Motif.objects.filter(document = document).order_by('-'+score_type)
    docfeatures = DocumentGlobalFeature.objects.filter(document = document)
    # Add the parent data
    score = "na"
    top_motifs = []
    if not motif == None:
        topdm2m = DocumentGlobalMass2Motif.objects.get(document = document,mass2motif = motif)
        top_motifs.append(topdm2m.mass2motif)
        if score_type == 'probability':
            score = topdm2m.probability
        else:
            score = topdm2m.overlap_score
    else:
        for dm in docm2m:
            top_motifs.append(dm.mass2motif)
            if len(top_motifs) == len(colours):
                break

    parent_data = (document.mass, 100.0, document.display_name, document.annotation, score)
    plot_fragments = []
    plot_data.append(parent_data)


    # get the max intensity for normalisation
    maxi = 0.0
    for docfeature in docfeatures:
        if docfeature.intensity > maxi:
            maxi = docfeature.intensity

    for docfeature in docfeatures:
        # Find the motifs for this feature
        if docfeature.feature.name.startswith('fragment'):
            intensity = docfeature.intensity
            cum_intensity = 0.0
            mz = float(docfeature.feature.name.split('_')[1])
            for dm2m in docm2m:
                dfm2ms = DocumentFeatureMass2Motif.objects.filter(docm2m = dm2m,feature = docfeature.feature)
                if len(dfm2ms) > 0:
                    if dm2m.mass2motif in top_motifs:
                        colour = colours[top_motifs.index(dm2m.mass2motif)]
                    else:
                        colour = 'gray'
                    start_intensity = cum_intensity
                    for dfm2m in dfm2ms:
                        cum_intensity += intensity * dfm2m.probability
                    plot_fragments.append([mz,mz,
                                       100.0*start_intensity/(1.0*maxi),
                                       100.0*cum_intensity/(1.0*maxi),
                                       1,colour,
                                       docfeature.feature.name])
        elif docfeature.feature.name.startswith('loss'):
            intensity = docfeature.intensity
            yval = 100.0*intensity/(1.0*maxi)
            total_width = float(docfeature.feature.name.split('_')[1])
            start_x = document.mass - total_width
            cum_x = start_x
            for dm2m in docm2m:
                dfm2ms = DocumentFeatureMass2Motif.objects.filter(docm2m = dm2m,feature = docfeature.feature)
                if len(dfm2ms) > 0:
                    if dm2m.mass2motif in top_motifs:
                        colour = colours[top_motifs.index(dm2m.mass2motif)]
                    else:
                        colour = 'gray'

                    start_x = cum_x

                    for dfm2m in dfm2ms:
                        cum_x += total_width * dfm2m.probability

                    plot_fragments.append([start_x,
                                           cum_x,
                                           yval,
                                           yval,
                                           0,
                                           colour,
                                           docfeature.feature.name])
    plot_data.append(plot_fragments)
    if get_key:
        key = []
        for i,topic in enumerate(top_motifs):
            if topic.originalmotif.annotation:
                topic_info = topic.originalmotif.name + " (" + topic.originalmotif.annotation + ")"
            else:
                topic_info = topic.originalmotif.name
            key.append((topic_info ,colours[i]))
        return [plot_data],key
    
    return plot_data


def get_decomp_doc_context_dict(document):
    context_dict = {}
    features = DocumentGlobalFeature.objects.filter(document = document)
    context_dict['features'] = features
    dm2m = DocumentGlobalMass2Motif.objects.filter(document = document)
    context_dict['mass2motifs'] = dm2m
    feature_mass2motif_instances = []
    for feature in features:
        if feature.intensity > 0:
            feature_mass2motif_instances.append((feature,
                DocumentFeatureMass2Motif.objects.filter(docm2m__document = document,
                                                      feature = feature.feature)))
    feature_mass2motif_instances = sorted(feature_mass2motif_instances,
                                          key = lambda x:x[0].intensity,
                                                reverse = True)
    context_dict['fm2m'] = feature_mass2motif_instances
    return context_dict