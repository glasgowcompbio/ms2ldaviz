import numpy as np
import bisect
import jsonpickle
from scipy.special import psi as psi
from scipy.special import polygamma as pg
from scipy.sparse import coo_matrix
import networkx as nx
from networkx.readwrite import json_graph
from decomposition.models import DocumentGlobalFeature,GlobalFeature,GlobalMotif,DocumentGlobalMass2Motif,DocumentFeatureMass2Motif,FeatureSet,Decomposition,FeatureMap,Beta
from basicviz.models import VizOptions,Experiment,Document,Mass2MotifInstance
from annotation.models import TaxaInstance,SubstituentInstance
from options.views import get_option
from ms1analysis.models import Sample, DocSampleIntensity
from ms1analysis.models import DecompositionAnalysis, DecompositionAnalysisResult, DecompositionAnalysisResultPlage

import sys
sys.path.append('../lda/code')
from ms2lda_feature_extraction import LoadMZML, LoadMSP, LoadMGF

def add_sample(sample_name, experiment):
    sample = Sample.objects.get_or_create(name = sample_name, experiment = experiment)[0]
    return sample

def add_doc_sample_intensity(sample, document, intensity):
    i = DocSampleIntensity.objects.get_or_create(sample = sample, document = document, intensity = intensity)[0]

def load_sample_intensity(document, experiment, metadata):
    # metadata = lda_dict['doc_metadata'][doc_name]
    if 'intensities' in metadata:
        for sample_name, intensity in metadata['intensities'].items():
            ## process missing data
            ## if intensity not exist, does not save in database
            if intensity:
                sample = add_sample(sample_name, experiment)
                add_doc_sample_intensity(sample, document, intensity)

def load_mzml_and_make_documents(experiment,motifset):
    assert experiment.ms2_file
    peaklist = None
    if experiment.csv_file:
        peaklist = experiment.csv_file.path

    if experiment.experiment_ms2_format == '0':
        loader = LoadMZML(isolation_window=experiment.isolation_window, mz_tol=experiment.mz_tol,
                          rt_tol=experiment.rt_tol, peaklist=peaklist,
                          min_ms1_intensity = experiment.min_ms1_intensity,
                          duplicate_filter = experiment.filter_duplicates,
                          duplicate_filter_mz_tol = experiment.duplicate_filter_mz_tol,
                          duplicate_filter_rt_tol = experiment.duplicate_filter_rt_tol,
                          min_ms1_rt = experiment.min_ms1_rt,
                          max_ms1_rt = experiment.max_ms1_rt,
                          min_ms2_intensity = experiment.min_ms2_intensity)
    elif experiment.experiment_ms2_format == '1':
        loader = LoadMSP(min_ms1_intensity = experiment.min_ms1_intensity,
                        min_ms2_intensity = experiment.min_ms2_intensity,
                        mz_tol=experiment.mz_tol,
                        rt_tol=experiment.rt_tol,
                        peaklist=peaklist)
    elif experiment.experiment_ms2_format == '2':
        loader = LoadMGF(min_ms1_intensity = experiment.min_ms1_intensity,
                        min_ms2_intensity = experiment.min_ms2_intensity,
                        mz_tol=experiment.mz_tol,
                        rt_tol=experiment.rt_tol,
                        peaklist=peaklist)

    print("Loading peaks from {} using peaklist {}".format(experiment.ms2_file.path,peaklist))
    ms1,ms2,metadata = loader.load_spectra([experiment.ms2_file.path])
    print("Loaded {} MS1 peaks and {} MS2 peaks".format(len(ms1),len(ms2)))

    # feature set and original experiment hardcoded for now
    fs = motifset.featureset

    # decompose_from is the original experiment name, e.g. 'massbank_binned_005'
    # TODO: what to do with this?
    # original_experiment = Experiment.objects.get(name=decompose_from)

    features = GlobalFeature.objects.filter(featureset=fs).order_by('min_mz')

    fragment_features = [f for f in features if f.name.startswith('fragment')]
    loss_features = [f for f in features if f.name.startswith('loss')]
    min_frag_mz = [f.min_mz for f in fragment_features]
    max_frag_mz = [f.max_mz for f in fragment_features]
    min_loss_mz = [f.min_mz for f in loss_features]
    max_loss_mz = [f.max_mz for f in loss_features]

    # Delete any already existing docs (mainly for debugging)
    # docs = Document.objects.filter(experiment = experiment)
    # print("Found {} documents to delete".format(len(docs)))
    # for doc in docs:
    #     doc.delete()

    # Add the documents to the database
    n_done = 0
    n_new_features = 0
    for molecule in ms1:
        ms2_features = filter(lambda x:x[3]==molecule,ms2)
        if len(ms2_features) == 0:
            continue
        name = molecule.name + '_decomp'
        new_doc,status = Document.objects.get_or_create(experiment = experiment,name = name)
        ## load peaklist (sample names) to database
        load_sample_intensity(new_doc, experiment, metadata[molecule.name])
        doc_metadata = {}
        doc_metadata['parentmass'] = molecule.mz
        doc_metadata['parentrt'] = molecule.rt
        new_doc.metadata = jsonpickle.encode(doc_metadata)
        new_doc.save()
        for f in ms2_features:
            fragment_mz = f[0]
            intensity = f[2]
            frag_pos = bisect.bisect_right(min_frag_mz,fragment_mz)-1
            if fragment_mz <= max_frag_mz[frag_pos]:
                feat = fragment_features[frag_pos]
                df,status = DocumentGlobalFeature.objects.get_or_create(document = new_doc,feature = feat)
                if status:
                    df.intensity = intensity
                else:
                    df.intensity += intensity
                df.save()
            else:
                # make a new feature
                # Assumes binned_005 featureset
                tempmz = fragment_mz*100
                if tempmz - np.floor(tempmz) > 0.5:
                    min_mz = np.floor(tempmz)/100 + 0.005
                else:
                    min_mz = np.floor(tempmz)/100
                max_mz = min_mz + 0.005
                new_feature_name = 'fragment_{}'.format((max_mz + min_mz)/2.0)
                gf,status = GlobalFeature.objects.get_or_create(max_mz = max_mz,min_mz = min_mz,name = new_feature_name,featureset = fs)
                n_new_features += 1
                df,status = DocumentGlobalFeature.objects.get_or_create(document = new_doc,feature = gf)
                if status:
                    df.intensity = intensity
                else:
                    df.intensity += intensity
                df.save()
                
            loss_mz = molecule.mz - fragment_mz
            if loss_mz >= min_loss_mz[0] and loss_mz <= max_loss_mz[-1]:
                loss_pos = bisect.bisect_right(min_loss_mz,loss_mz)-1
                if loss_mz <= max_loss_mz[loss_pos]:
                    feat = loss_features[loss_pos]
                    df,status = DocumentGlobalFeature.objects.get_or_create(document = new_doc,feature  = feat)
                    if status:
                        df.intensity = intensity
                    else:
                        df.intensity += intensity
                    df.save()
                else:
                    # make a new feature
                    tempmz = loss_mz*100
                    if tempmz - np.floor(tempmz) > 0.5:
                        min_mz = np.floor(tempmz)/100 + 0.005
                    else:
                        min_mz = np.floor(tempmz)/100
                    max_mz = min_mz + 0.005
                    new_feature_name = 'loss_{}'.format((max_mz + min_mz)/2.0)
                    gf,status = GlobalFeature.objects.get_or_create(max_mz = max_mz,min_mz = min_mz,name = new_feature_name,featureset = fs)
                    n_new_features += 1
                    df,status = DocumentGlobalFeature.objects.get_or_create(document = new_doc,feature = gf)
                    if status:
                        df.intensity = intensity
                    else:
                        df.intensity += intensity
                    df.save()
                    
        n_done += 1
        if n_done % 100 == 0:
            print("Done {} documents (required {} new features)".format(n_done,n_new_features))




def decompose(decomposition,normalise = 1000.0,store_threshold = 0.01):
    motifset = decomposition.motifset
    betaobject = Beta.objects.get(motifset = motifset)
    documents = Document.objects.filter(experiment = decomposition.experiment)
    print("Loading and unpickling beta")

    alpha = jsonpickle.decode(betaobject.alpha_list)
    motif_id_list = jsonpickle.decode(betaobject.motif_id_list)
    feature_id_list = jsonpickle.decode(betaobject.feature_id_list)
    beta = jsonpickle.decode(betaobject.beta)

    n_motifs = len(motif_id_list)
    n_features = len(feature_id_list)

    # if decomposition.motifset.name.startswith('gnps'):
        # assuming sparse beta
        # Naive construction
        # beta_matrix = np.zeros((n_motif,n_feature),np.float)
        # for r,c,v in beta:
        #     beta_matrix[r,c] = v
        # or using sparse...
    r,c,data = zip(*beta)
    beta_matrix = np.array(coo_matrix((data,(r,c)),shape=(n_motifs,n_features)).todense())
    s = beta_matrix.sum(axis=1)[:,None]
    s[s==0] = 1.0
    beta_matrix /= s # makes beta a full matrix. Note that can keep it sparse by beta_matrix.data /= s[beta_matrix.row]
    
        # beta_matrix = coo.todense()
    # else:
        # beta_matrix = np.array(beta)


    alpha_matrix = np.array(alpha)
    

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

    g_term = np.zeros(K)
    print("Performing e-steps")
    total_docs = len(documents)
    for i in range(total_docs):
        document = documents[i]
        print('%d/%d: %s' % (i, total_docs, document.name))
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
            # print("Iteration {}".format(i))
            temp_gamma = np.zeros(K) + alpha_matrix
            for word,intensity in doc_dict.items():
                # Find the word position in beta
                if word in word_index:
                    word_pos = word_index[word]
                    # beta_col = beta_matrix.getcol(word_pos).todense().flatten()
                    # if beta_col.sum() > 0:
                    if beta_matrix[:,word_pos].sum() > 0:
                        log_phi_matrix = np.log(beta_matrix[:,word_pos]) + psi(gamma)
                        # log_phi_matrix = np.log(beta_col) + psi(gamma).T
                        log_phi_matrix = np.exp(log_phi_matrix - log_phi_matrix.max())
                        phi_matrix[word] = log_phi_matrix/log_phi_matrix.sum()
                        temp_gamma += phi_matrix[word]*intensity

            gamma = temp_gamma.copy()
        g_term += psi(gamma) - psi(gamma.sum())
        
        
        # normalise the gamma to get probabilities
        theta = gamma/gamma.sum()
        theta = list(theta.flatten())

        theta_motif = zip(theta,motif_list)
        theta_motif = sorted(theta_motif,key = lambda x : x[0],reverse = True)
        theta,motif = zip(*theta_motif)
        tot_prob = 0.0
        for i in range(K):
            if theta[i] < store_threshold:
                break
            motif_pos = motif_index[motif[i]]
            overlap_score = compute_overlap(phi_matrix,motif_pos,beta_matrix[motif_pos,:],word_index)
            dgm2m,status = DocumentGlobalMass2Motif.objects.get_or_create(document = document,mass2motif = motif[i],decomposition=decomposition)
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
    M = total_docs
    alpha = alpha_nr(g_term,M)


def compute_overlap(phi_matrix,motif_pos,beta_row,word_index):
    overlap_score = 0.0
    for word in phi_matrix:
        if word in word_index:
            word_pos = word_index[word]
            if phi_matrix[word] == None:
                continue
            else:
                overlap_score += phi_matrix[word][motif_pos]*beta_row[word_pos]
    return overlap_score


def get_parents_decomposition(motif_id,decomposition,experiment = None):
    # if vo_id:
    #     viz_options = VizOptions.objects.get(id = vo_id)
    #     edge_choice = viz_options.edge_choice
    #     edge_thresh = viz_options.edge_thresh
    # elif experiment:
    #     edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    #     edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)
    # else:
    #     edge_choice = 'probability'
    #     edge_thresh = 0.05
    motif = GlobalMotif.objects.get(id = motif_id)
    parent_data = []
    docm2m = get_docglobalm2m(motif, decomposition)
    # if edge_choice == 'probability':
    #     docm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh,decomposition = decomposition).order_by('-probability')
    # elif edge_choice == 'overlap_score':
    #     docm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,decomposition = decomposition).order_by('-overlap_score')
    # elif edge_choice == 'both':
    #     docm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,probability__gte = edge_thresh,decomposition = decomposition).order_by('-overlap_score')
    for dm in docm2m:
        document = dm.document
        parent_data.append(get_parent_for_plot_decomp(decomposition,document,motif = motif))

    return parent_data


## this function is used to get data for preperation of spectrum plot for decomposition experiments
## need to construct *plot_fragments* list, which is [parent_data, child_data]
## parent_data (tuple)
## child_data (list of tuples)
## we show both probability and overlap scores in title of plot
## and choose topics with highest probability scores (up to 6) when colouring
def get_parent_for_plot_decomp(decomposition,document,motif = None,get_key = False):
    plot_data = []
    colours = ['red', 'green', 'black', 'yellow', 'purple', 'silver']
    # edge_order = edge_choice
    # if edge_choice == 'both':
    #     edge_order = 'probability'

    # Add the parent data
    score = "na"
    top_motifs = []
    if not motif == None:
        topdm2m = DocumentGlobalMass2Motif.objects.get(decomposition = decomposition,document = document,mass2motif = motif)
        top_motifs.append(topdm2m.mass2motif)
        probability = topdm2m.probability
        overlap_score = topdm2m.overlap_score
        # if edge_choice == 'probability':
        #     score = topdm2m.probability
        # else:
        #     score = topdm2m.overlap_score


    parent_data = (document.mass, 100.0, document.display_name, document.annotation, probability, overlap_score)
    plot_fragments = []
    plot_data.append(parent_data)

    ## prepare child data
    docm2m = DocumentGlobalMass2Motif.objects.filter(decomposition = decomposition,document = document).order_by('-probability')
    docfeatures = DocumentGlobalFeature.objects.filter(document = document)

    if motif == None:
        for dm in docm2m:
            top_motifs.append(dm.mass2motif)
            if len(top_motifs) == len(colours):
                break
    # get the max intensity for normalisation
    maxi = 0.0
    for docfeature in docfeatures:
        if docfeature.intensity > maxi:
            maxi = docfeature.intensity

    for docfeature in docfeatures:
        # Find the motifs for this feature
        if docfeature.feature.name.startswith('fragment'):
            plotted = False
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
                    plotted = True
            if not plotted:
                # This feature wasnt decomposed - plot it grey
                mz = float(docfeature.feature.name.split('_')[1])
                colour = 'gray'
                intensity = 100.0*docfeature.intensity/(1.0*maxi)
                plot_fragments.append([mz,mz,0,intensity,1,colour,docfeature.feature.name])

        elif docfeature.feature.name.startswith('loss'):
            plotted = False
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
                    plotted = True

            if not plotted:
                colour = 'gray'
                plot_fragments.append([start_x,start_x+total_width,yval,yval,0,colour,docfeature.feature.name])

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


# Get the context dictionary for displaying a document
def get_decomp_doc_context_dict(decomposition,document):
    context_dict = {}
    features = DocumentGlobalFeature.objects.filter(document = document)
    context_dict['features'] = features
    dm2m = DocumentGlobalMass2Motif.objects.filter(decomposition = decomposition,document = document)
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



def make_word_graph(request, motif_id, vo_id, decomposition_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    # if not vo_id == 'nan':
    #     viz_options = VizOptions.objects.get(id = vo_id)
    #     experiment = viz_options.experiment
    #     edge_thresh = viz_options.edge_thresh
    #     edge_choice = viz_options.edge_choice
    # elif experiment:
    #     edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    #     edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)
    # else:
    #     edge_choice = 'probability'
    #     edge_thresh = 0.05
 
    data_for_json = []
    motif = GlobalMotif.objects.get(id = motif_id)
    originalmotif = motif.originalmotif
    originalfeatures = Mass2MotifInstance.objects.filter(mass2motif = originalmotif,probability__gte = 0.01)
    globalfeatures = FeatureMap.objects.filter(localfeature__in = [o.feature for o in originalfeatures])
    globalfeatures = [g.globalfeature for g in globalfeatures]
    # if edge_choice == 'probability':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh,decomposition = decomposition)
    # elif edge_choice == 'overlap_score':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,decomposition = decomposition)
    # elif edge_choice == 'both':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,probability__gte = edge_thresh,decomposition = decomposition)
    # else:
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh,decomposition = decomposition)
    
    docm2ms = get_docglobalm2m(globalm2m = motif, decomposition = decomposition)
    data_for_json.append(len(docm2ms))
    feat_counts = {}
    for feature in globalfeatures:
        feat_counts[feature] = 0
    for dm2m in docm2ms:
        fi = DocumentGlobalFeature.objects.filter(document = dm2m.document)
        for ft in fi:
            if ft.feature in feat_counts:
                feat_counts[ft.feature] += 1
    colours = '#404080'
    feat_list = []
    for feature in feat_counts:
        feat_type = feature.name.split('_')[0]
        feat_mz = feature.name.split('_')[1]
        short_name = "{}_{:.4f}".format(feat_type,float(feat_mz))
        feat_list.append([short_name,feat_counts[feature],colours])
    feat_list = sorted(feat_list,key = lambda x: x[1],reverse = True)
    data_for_json.append(feat_list)

    return data_for_json

def make_intensity_graph(request, motif_id, vo_id, decomposition_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id = vo_id)
        experiment = viz_options.experiment
    #     edge_thresh = viz_options.edge_thresh
    #     edge_choice = viz_options.edge_choice
    # elif experiment:
    #     edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    #     edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)
    # else:
    #     edge_choice = 'probability'
    #     edge_thresh = 0.05

    colours = ['#404080', '#0080C0']
    colours = ['red','blue']



    data_for_json = []
    motif = GlobalMotif.objects.get(id = motif_id)
    originalmotif = motif.originalmotif
    originalfeatures = Mass2MotifInstance.objects.filter(mass2motif = originalmotif,probability__gte = 0.01)
    globalfeatures = FeatureMap.objects.filter(localfeature__in = [o.feature for o in originalfeatures])
    globalfeatures = [g.globalfeature for g in globalfeatures]
    # if edge_choice == 'probability':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh,decomposition = decomposition)
    # elif edge_choice == 'overlap_score':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,decomposition = decomposition)
    # elif edge_choice == 'both':
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh,probability__gte = edge_thresh,decomposition = decomposition)
    # else:
    #     docm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh,decomposition = decomposition)
    docm2ms = get_docglobalm2m(motif, decomposition)
    documents = [d.document for d in docm2ms]
    
    feat_total_intensity = {}
    feat_motif_intensity = {}
    for feature in globalfeatures:
        feat_total_intensity[feature] = 0.0
        feat_motif_intensity[feature] = 0.0
    for feature in globalfeatures:
        fi = DocumentGlobalFeature.objects.filter(document__experiment = experiment,feature = feature)
        for ft in fi:
            feat_total_intensity[feature] += ft.intensity
            if ft.document in documents:
                feat_motif_intensity[feature] += ft.intensity
    

    feat_list = []
    feat_tot_intensity = zip(feat_total_intensity.keys(),feat_total_intensity.values())
    feat_tot_intensity = sorted(feat_tot_intensity,key = lambda x: x[1],reverse = True)
    for feature,tot_intensity in feat_tot_intensity:
        feat_type = feature.name.split('_')[0]
        feat_mz = feature.name.split('_')[1]
        short_name = "{}_{:.4f}".format(feat_type,float(feat_mz))
        feat_list.append([short_name,feat_total_intensity[feature],colours[0]])
        feat_list.append(['',feat_motif_intensity[feature],colours[1]])
        feat_list.append(('', 0, ''))
    data_for_json.append(feat_tot_intensity[0][1])
    data_for_json.append(feat_list)

    return data_for_json


## This function is basically a replicate from *make_graph* function of basicviz/views/views_lda_single.py
## For MS1 analysis, LDA and Decomposition both use *Sample* and *DocSampleIntensity* to store data from peaklist
## Differnece:
## Decomposition:
##      need *decomposition* input
##      use GlobalMotif linked to original motifs, use *DocumentGlobalMass2Motif* to get the relation
##      use DecompositionAnalysis, DecompositionAnalysisResult, DecompositionAnalysisResultPlage for MS1 analysis
##      use *get_docglobalm2m* function to get docm2ms result, notice motif is *GlobalMotif* here
## LDA:
##      only needs *experiment* input
##      use *DocumentMass2Motif* to get the relation
##      use Analysis, AnalysisResult, AnalysisResultPlage for MS1 analysis
##      use *get_docm2m* function to get docm2ms result
def make_decomposition_graph(decomposition,experiment,min_degree = 5,
                                topic_scale_factor = 5,
                                edge_scale_factor = 5,
                                ms1_analysis_id = None,
                                doc_max_size=200, motif_max_size=1000):
    # This is the graph maker for a decomposition experiment
    ## Notice mass2motif here is an object of GlobalMotif
    ## Get all unique Global mass2motifs, prepare for *get_docglobalm2m*
    all_docm2ms = DocumentGlobalMass2Motif.objects.filter(decomposition = decomposition)
    mass2motifs = set([docm2m.mass2motif for docm2m in all_docm2ms])

    # Find the degrees
    topics = {}
    docm2m_dict = {}
    for mass2motif in mass2motifs:
        topics[mass2motif] = 0
        docm2ms = get_docglobalm2m(mass2motif, decomposition)
        docm2m_dict[mass2motif] = list(docm2ms)

        for d in docm2ms:
            topics[mass2motif] += 1
    to_remove = []
    for topic in topics:
        if topics[topic] < min_degree:
            to_remove.append(topic)
    for topic in to_remove:
        del topics[topic]

    docm2mset = []
    for topic in topics:
        docm2mset += docm2m_dict[topic]


    do_plage_flag = True
    if ms1_analysis_id:
        analysis = DecompositionAnalysis.objects.filter(id=ms1_analysis_id)[0]
        all_logfc_vals = []
        res = DecompositionAnalysisResult.objects.filter(analysis=analysis, document__in=[docm2m.document for docm2m in docm2mset])
        for analysis_result in res:
            foldChange = analysis_result.foldChange
            logfc = np.log(foldChange)
            if not np.abs(logfc) == np.inf:
                all_logfc_vals.append(np.log(foldChange))
        min_logfc = np.min(all_logfc_vals)
        max_logfc = np.max(all_logfc_vals)

        ## try make graph for plage
        all_plage_vals = []
        for plage_result in DecompositionAnalysisResultPlage.objects.filter(analysis=analysis, globalmotif__in=topics.keys()):
            plage_t_value = plage_result.plage_t_value
            all_plage_vals.append(plage_t_value)
        if all_plage_vals:
            min_plage = np.min(all_plage_vals)
            max_plage = np.max(all_plage_vals)
        else:
            do_plage_flag = False

    print("First")
    # Add the topics to the graph
    G = nx.Graph()
    for topic in topics:
        mass2motif = topic.originalmotif
        metadata = jsonpickle.decode(mass2motif.metadata)
        ## try make graph for plage
        if ms1_analysis_id and do_plage_flag:
            ## white to green
            lowcol = [255, 255, 255]
            endcol = [0, 255, 0]
            plage_result = DecompositionAnalysisResultPlage.objects.filter(analysis=analysis, globalmotif=topic)[0]
            plage_t_value = plage_result.plage_t_value
            plage_p_value = plage_result.plage_p_value
            pos = (plage_t_value - min_plage) / (max_plage - min_plage)
            r = lowcol[0] + int(pos * (endcol[0] - lowcol[0]))
            g = lowcol[1] + int(pos * (endcol[1] - lowcol[1]))
            b = lowcol[2] + int(pos * (endcol[2] - lowcol[2]))
            col = "#{}{}{}".format("{:02x}".format(r), "{:02x}".format(g), "{:02x}".format(b))
            if plage_p_value == None:
                size = 10
            elif plage_p_value == 0:
                size = motif_max_size
            else:
                size = min(10 - np.log(plage_p_value) * 200, motif_max_size)
            na = mass2motif.short_annotation
            if na:
                na += ' (' + topic.name + ')'
            else:
                na = topic.name
            G.add_node(topic.name, group=2, name=na+", "+str(plage_t_value) + ", "+str(plage_p_value),
                       # size=topic_scale_factor * topics[topic],
                       size= size,
                       special=True, in_degree=topics[topic],
                       highlight_colour=col,
                       score=1, node_id=topic.id, is_topic=True)

        else:
            if mass2motif.short_annotation:
            # if 'annotation' in metadata:
                G.add_node(topic.name, group=2, name=mass2motif.short_annotation,
                           size=topic_scale_factor * topics[topic],
                           special=True, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)
            else:
                G.add_node(topic.name, group=2, name=topic.name,
                           size=topic_scale_factor * topics[topic],
                           special=False, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)

    doc_nodes = []

    print("Second")

    ## do this change, because in LDA plot, it has fixed the edge_choice to be 'probability'
    # edge_choice = get_option('default_doc_m2m_score', experiment)
    edge_choice = 'probability'

    for docm2m in docm2mset:
        # if docm2m.mass2motif in topics:
        if not docm2m.document in doc_nodes:
            metadata = jsonpickle.decode(docm2m.document.metadata)
            if 'compound' in metadata:
                name = metadata['compound']
            elif 'annotation' in metadata:
                name = metadata['annotation']
            else:
                name = docm2m.document.name
            ## do MS1 expression analysis only when user choose a ms1 analysis setting
            if not ms1_analysis_id:
                G.add_node(docm2m.document.name, group=1, name=name, size=20,
                           type='square', peakid=docm2m.document.name, special=False,
                           in_degree=0, score=0, is_topic=False)
            else:
                analysis_result = DecompositionAnalysisResult.objects.filter(analysis=analysis, document=docm2m.document)[0]
                foldChange = analysis_result.foldChange
                pValue = analysis_result.pValue
                logfc = np.log(foldChange)

                ## lowest: blue, logfc==0: white, highest: red
                ## use scaled colour to represent logfc of document
                if logfc == np.inf:
                    col = "#{}{}{}".format('FF', '00', '00')
                elif -logfc == np.inf:
                    col = "#{}{}{}".format('00', '00', 'FF')
                else:
                    lowcol = [0, 0, 255]
                    endcol = [255, 0, 0]
                    midcol = [255, 255, 255]
                    if logfc < 0:
                        # if logfc < -3:
                        #     logfc = -3
                        pos = logfc/ min_logfc
                        r = midcol[0] + int(pos * (lowcol[0] - midcol[0]))
                        g = midcol[1] + int(pos * (lowcol[1] - midcol[1]))
                        b = midcol[2] + int(pos * (lowcol[2] - midcol[2]))
                    else:
                        pos = logfc / max_logfc
                        r = midcol[0] + int(pos * (endcol[0] - midcol[0]))
                        g = midcol[1] + int(pos * (endcol[1] - midcol[1]))
                        b = midcol[2] + int(pos * (endcol[2] - midcol[2]))
                    col = "#{}{}{}".format("{:02x}".format(r), "{:02x}".format(g), "{:02x}".format(b))

                ## use size to represent pValue of document
                if not pValue:
                    size = 5
                else:
                    size = min(5 - np.log(pValue) * 15, doc_max_size)
                ## represent document node with name + logfc + pValue
                if pValue:
                    name = "{}, {:.3f}, {:.3f}".format(name, logfc, pValue)
                else:
                    name = "{}, {:.3f}, None".format(name, logfc)
                # name += ", " + str(logfc) + ", " + str(pValue)
                G.add_node(docm2m.document.name, group=1, name=name, size=size,
                           type='square', peakid=docm2m.document.name, special=True,
                           highlight_colour=col, logfc=docm2m.document.logfc,
                           in_degree=0, score=0, is_topic=False)

            doc_nodes.append(docm2m.document)


        if edge_choice == 'probability':
            weight = edge_scale_factor * docm2m.probability
        elif edge_choice == 'both':
            weight = docm2m.overlap_score
        else:
            weight = edge_scale_factor * docm2m.overlap_score
        G.add_edge(docm2m.mass2motif.name, docm2m.document.name, weight=weight)
    print("Third")

    return G
        


def api_decomposition(doc_dict,motifset):

    normalise = 1000.0

    betaobject = Beta.objects.get(motifset = motifset)
    alpha = jsonpickle.decode(betaobject.alpha_list)
    motif_id_list = jsonpickle.decode(betaobject.motif_id_list)
    feature_id_list = jsonpickle.decode(betaobject.feature_id_list)
    beta = jsonpickle.decode(betaobject.beta)


    n_motifs = len(motif_id_list)
    n_features = len(feature_id_list)

    r,c,data = zip(*beta)
    beta_matrix = np.array(coo_matrix((data,(r,c)),shape=(n_motifs,n_features)).todense())
    s = beta_matrix.sum(axis=1)[:,None]
    s[s==0] = 1.0
    beta_matrix /= s # makes beta a full matrix. Note that can keep it sparse by beta_matrix.data /= s[beta_matrix.row]


    alpha_matrix = np.array(alpha)
    

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
    print("Performing e-steps")
    total_docs = len(doc_dict)

    results = {}
    results['decompositions'] = {}
    results['terms'] = {}
    g_term = np.zeros(K)

    for i,doc in enumerate(doc_dict.keys()):
        results['decompositions'][doc] = []
        results['terms'][doc] = []
        document = doc_dict[doc]
        print('%d/%d: %s' % (i, total_docs, doc))

        maxi = 0.0
        for feature,intensity in doc_dict[doc].items():
            if intensity > maxi:
                maxi = intensity

        # normalise
        if normalise:
            for word in doc_dict[doc]:
                doc_dict[doc][word] = int(normalise*doc_dict[doc][word]/maxi)

        # Do the e-steps for this document
        phi_matrix = {}
        for word in doc_dict[doc]:
            phi_matrix[word] = None
        gamma = np.ones(K)
        for ei in range(100): # do 20 iterations
            # print("Iteration {}".format(ei))
            temp_gamma = np.zeros(K) + alpha_matrix
            # temp_gamma = np.ones_like(alpha_matrix)
            for word,intensity in doc_dict[doc].items():
                # Find the word position in beta
                if word in word_index:
                    word_pos = word_index[word]
                    if beta_matrix[:,word_pos].sum() > 0:
                        log_phi_matrix = np.log(beta_matrix[:,word_pos]) + psi(gamma)
                        log_phi_matrix = np.exp(log_phi_matrix - log_phi_matrix.max())
                        phi_matrix[word] = log_phi_matrix/log_phi_matrix.sum()
                        temp_gamma += phi_matrix[word]*intensity

            gamma = temp_gamma.copy()
        g_term += psi(gamma) - psi(gamma.sum())        
        
        # normalise the gamma to get probabilities
        theta = gamma/gamma.sum()
        theta = list(theta.flatten())

        theta_motif = zip(theta,motif_list)
        theta_motif = sorted(theta_motif,key = lambda x : x[0],reverse = True)
        pos = 0
        cum_prob = 0.0

        sub_terms = {}
        tax_terms = {}

        while cum_prob < 0.99:
            theta,motif = theta_motif[pos]
            motif_pos = motif_index[motif]
            overlap_score = compute_overlap(phi_matrix,motif_pos,beta_matrix[motif_pos,:],word_index)


            tax_term_instances = TaxaInstance.objects.filter(motif = motif.originalmotif)
            sub_term_instances = SubstituentInstance.objects.filter(motif = motif.originalmotif)

            for t in tax_term_instances:
                if not t.taxterm in tax_terms:
                    tax_terms[t.taxterm] = 0.0
                tax_terms[t.taxterm] += overlap_score * t.probability
            for s in sub_term_instances:
                if not s.subterm in sub_terms:
                    sub_terms[s.subterm] = 0.0
                sub_terms[s.subterm] += overlap_score * s.probability

            results['decompositions'][doc].append((motif.name,motif.originalmotif.name,theta,overlap_score,motif.originalmotif.annotation))

            cum_prob += theta
            pos += 1

        for s in sub_terms:
            if sub_terms[s] >= 0.5:
                results['terms'][doc].append((s.name,'substituent',sub_terms[s]))
        for t in tax_terms:
            if tax_terms[t] >= 0.5:
                results['terms'][doc].append((t.name,'taxa',tax_terms[t]))


    # Do the alpha optimisation
    if total_docs > 1:
        M = len(doc_dict)
        alpha = alpha_nr(g_term,M)
        alpha_list = []
        for a in alpha:
            alpha_list.append(float(a))
    else:
        alpha_list = []
    results['alpha'] = zip([m.name for m in motif_list],alpha_list)
    results['motifset'] = motifset.name
    return results


def make_documents(spectra,featureset):
    import bisect
    features = GlobalFeature.objects.filter(featureset = featureset).order_by('min_mz')
    fragment_features = [f for f in features if f.name.startswith('fragment')]
    loss_features = [f for f in features if f.name.startswith('loss')]
    min_frag_mz = fragment_features[0].min_mz
    max_frag_mz = fragment_features[-1].max_mz

    min_loss_mz = loss_features[0].min_mz
    max_loss_mz = loss_features[-1].max_mz

    min_frag_mz_list = [f.min_mz for f in fragment_features]
    min_loss_mz_list = [f.min_mz for f in loss_features]

    doc_dict = {}
    doc_id = 0
    for doc_name,parentmass,peaks in spectra:
        doc_dict[doc_name] = {}
        doc_id += 1
        for peak in peaks:
            fragment_mz = peak[0]
            intensity = peak[1]
            loss_mz = parentmass - fragment_mz
            feature = None
            if fragment_mz >= min_frag_mz and fragment_mz <= max_frag_mz:
                fragment_pos = bisect.bisect_right(min_frag_mz_list,fragment_mz) - 1
                if fragment_mz <= fragment_features[fragment_pos].max_mz:
                    feature = fragment_features[fragment_pos]
            if feature:
                if not feature in doc_dict[doc_name]:
                    doc_dict[doc_name][feature] = intensity
                else:
                    doc_dict[doc_name][feature] +=intensity
            feature = None
            if loss_mz >= min_loss_mz and loss_mz <= max_loss_mz:
                loss_pos = bisect.bisect_right(min_loss_mz_list,loss_mz) - 1
                if loss_mz <= loss_features[loss_pos].max_mz:
                    feature = loss_features[loss_pos]
            if feature:
                if not feature in doc_dict[doc_name]:
                    doc_dict[doc_name][feature] = intensity
                else:
                    doc_dict[doc_name][feature] +=intensity

    return doc_dict



def alpha_nr(g_term,M,maxit=100,init_alpha=[]):

    # with open('./test_g.csv','w') as f:
    #     for g in g_term:
    #         f.write("{}\n".format(g))

    SMALL_NUMBER = 1e-100
    K = len(g_term)
    if len(init_alpha) == 0:
        init_alpha = np.ones_like(g_term)/K
    old_alpha = init_alpha.copy()
    
    
    # try:
    alpha = init_alpha.copy()
    # g_term = (psi(self.gamma_matrix) - psi(self.gamma_matrix.sum(axis=1))[:,None]).sum(axis=0)
    for it in range(maxit):
        grad = M * (psi(alpha.sum()) - psi(alpha)) + g_term
        # H = -M*np.diag(pg(1,alpha)) + M*pg(1,alpha.sum())
        z = M*pg(1,alpha.sum())
        h = -M*pg(1,alpha)
        c = ((grad/h).sum())/((1.0/z) + (1.0/h).sum())
        alpha_change = (grad - c)/h

        # Check to make sure none of them go negative
        n_bad = (alpha_change > alpha).sum()
        while n_bad > 0:
            alpha_change/=2.0
            n_bad = (alpha_change > alpha).sum()


        # alpha_change = np.dot(np.linalg.inv(H),grad)
        alpha_new = alpha - alpha_change

        pos = np.where(alpha_new <= SMALL_NUMBER)[0]
        alpha_new[pos] = SMALL_NUMBER

        diff = np.sum(np.abs(alpha-alpha_new))
        # print("Alpha: {}, it: {}".format(diff,it))
        # print(grad.max(),grad.argmax(),alpha[grad.argmax()],alpha_new[grad.argmax()],alpha_change[grad.argmax()],h[grad.argmax()])
        alpha = alpha_new
        
        if diff < 1e-6 and it > 10:
            return alpha
    # except:
    #     alpha = old_alpha
    return alpha

def parse_spectrum_string(spectrum_string):
    # Parse the spectrum that has been input
    peaks = []
    tokens = spectrum_string.split()
    mz = None
    intensity = None
    for token in tokens:
        # First check for MONA format
        if ':' in token:
            vals = token.split(':')
            mz = float(vals[0])
            intensity = float(vals[1])
            peaks.append((mz,intensity))
            continue
        else:
            # If not MONA, assume that its just mz, rt pairs in a long list
            if mz is None:
                # Must be a new peak
                mz = float(token)
            elif intensity is None:
                # Already have a mz so this must be an intensity
                intensity = float(token)
                # Store the peak and then forget the mz and intensity
                peaks.append((mz,intensity))
                mz = None
                intensity = None
    return peaks

def get_docglobalm2m(globalm2m, decomposition, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    # experiment = mass2motif.experiment
    experiment = Experiment.objects.get(pk=decomposition.experiment_id)
    ## default prob_threshold 0.05, default overlap_threshld 0.0
    if not doc_m2m_prob_threshold:
        doc_m2m_prob_threshold = get_option('doc_m2m_prob_threshold', experiment = experiment)
        if doc_m2m_prob_threshold:
            doc_m2m_prob_threshold = float(doc_m2m_prob_threshold)
        else:
            doc_m2m_prob_threshold = 0.05

    if not doc_m2m_overlap_threshold:
        doc_m2m_overlap_threshold = get_option('doc_m2m_overlap_threshold', experiment = experiment)
        if doc_m2m_overlap_threshold:
            doc_m2m_overlap_threshold = float(doc_m2m_overlap_threshold)
        else:
            doc_m2m_overlap_threshold = 0.0

    ## Notice need to add *decomposition* when search database for *DocumentGlobalMass2Motif*
    ## For LDA experiment,  *DocumentMass2Motif* does not have *experiment* entry, so does not need to do that
    dm2m = DocumentGlobalMass2Motif.objects.filter(mass2motif=globalm2m, decomposition=decomposition, probability__gte=doc_m2m_prob_threshold,
                                                 overlap_score__gte=doc_m2m_overlap_threshold).order_by('-probability')

    return dm2m
