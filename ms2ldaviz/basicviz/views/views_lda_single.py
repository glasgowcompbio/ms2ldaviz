import csv
import json

import jsonpickle
import networkx as nx
import numpy as np
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse,Http404
from django.shortcuts import render,redirect
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA

from annotation.models import TaxaInstance,SubstituentInstance
from basicviz.forms import DocFilterForm, ValidationForm, VizForm, \
    TopicScoringForm, MatchMotifForm
from massbank.forms import Mass2MotifMetadataForm
from basicviz.models import Feature, Experiment, Document, FeatureInstance, DocumentMass2Motif, \
    FeatureMass2MotifInstance, Mass2Motif, Mass2MotifInstance, VizOptions, UserExperiment, MotifMatch
from basicviz.tasks import match_motifs
from massbank.views import get_massbank_form
from options.views import get_option
from decomposition.models import DocumentGlobalMass2Motif,GlobalMotif,DocumentGlobalFeature,FeatureMap
from decomposition.decomposition_functions import get_parents_decomposition,get_parent_for_plot_decomp,get_decomp_doc_context_dict
from basicviz.views import index as basicviz_index


def check_user(request,experiment):
    user = request.user
    try:
        ue = UserExperiment.objects.get(experiment = experiment,user = user)
        return ue.permission
    except: 
        # User can't see this one
        return None


def topic_table(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['motifs'] = motifs
    return render(request, 'basicviz/topic_table.html', context_dict)


def compute_topic_scores(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment': experiment}
    # Insert form logic here once form has been made
    # Start with a discrete hypergeo score
    documents = Document.objects.filter(experiment=experiment)
    md = jsonpickle.decode(documents[0].metadata)
    intensities = md['intensities']
    choices = [(i, i) for i in intensities.keys()]
    choices = sorted(choices, key=lambda x: x[0])

    if request.method == 'POST':
        form = TopicScoringForm(choices, request.POST)
        if form.is_valid():
            groups = form.cleaned_data['group1'] + form.cleaned_data['group2']

            intensities = []
            for document in documents:
                md = jsonpickle.decode(document.metadata)
                temp_intensity = []
                for group in groups:
                    temp_intensity.append(md['intensities'][group])
                intensities.append(temp_intensity)

            intensitynp = np.array(intensities)

            # Compute the logfc values
            logfc = []

            group1pos = range(len(form.cleaned_data['group1']))
            group2pos = range(len(form.cleaned_data['group1']), len(group1pos) + len(form.cleaned_data['group2']))

            SMALL = 1e-3
            for i, document in enumerate(documents):
                m1 = max(intensitynp[i, group1pos].mean(), SMALL)
                m2 = max(intensitynp[i, group2pos].mean(), SMALL)
                thisfc = np.log2(m1) - np.log2(m2)
                logfc.append(thisfc)
                if form.cleaned_data['storelogfc']:
                    md = jsonpickle.decode(document.metadata)
                    md['logfc'] = float(thisfc)
                    document.metadata = jsonpickle.encode(md)
                    document.save()

            logfc = np.array(logfc)

            # These should be set in a form
            lowperc = form.cleaned_data['lower_perc']
            upperc = form.cleaned_data['upper_perc']

            lfccopy = logfc.copy()
            lfccopy = np.sort(lfccopy)
            le = len(lfccopy)
            lowperc_value = lfccopy[int(np.floor(le * (lowperc / 100.0)))]
            upperc_value = lfccopy[int(np.ceil(le * (upperc / 100.0)))]

            total_above = len(np.where(logfc > upperc_value)[0])
            total_below = len(np.where(logfc < lowperc_value)[0])

            from scipy.stats.distributions import hypergeom

            M = len(documents)

            discrete_scores = []

            motifs = Mass2Motif.objects.filter(experiment=experiment)

            # doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = experiment)
            # if doc_m2m_threshold:
            #     doc_m2m_threshold = float(doc_m2m_threshold)
            # else:
            #     doc_m2m_threshold = 0.00 # Default value

            # default_score = get_option('default_doc_m2m_score',experiment = experiment)
            # if not default_score:
            #     default_score = 'probability'

            n_done = 0
            for motif1ind, motif in enumerate(motifs):
                score_list = []
                doc_indices = []

                # if default_score == 'probability':
                #     m2mdocs = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold)
                # else:
                # m2mdocs = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = doc_m2m_threshold)
                m2mdocs = get_docm2m(motif)
                for m2mdoc in m2mdocs:
                    doc_indices.append(list(documents).index(m2mdoc.document))
                n_above = 0
                n_below = 0
                for ind in doc_indices:
                    if logfc[ind] < lowperc_value:
                        n_below += 1
                    if logfc[ind] > upperc_value:
                        n_above += 1
                n_sub_docs = len(m2mdocs)
                up_range = range(n_above, n_sub_docs + 1)
                down_range = range(n_below, n_sub_docs + 1)
                score_list.append(n_sub_docs)
                score_list.append(n_above)
                up_score = hypergeom.pmf(up_range, M, total_above, n_sub_docs).sum()
                score_list.append(up_score)
                score_list.append(n_below)
                down_score = hypergeom.pmf(down_range, M, total_below, n_sub_docs).sum()
                score_list.append(down_score)
                discrete_scores.append((motif, score_list))

                if form.cleaned_data['savetopicscores']:
                    md = jsonpickle.decode(motif.metadata)
                    md['upscore'] = float(up_score)
                    md['downscore'] = float(down_score)
                    motif.metadata = jsonpickle.encode(md)
                    motif.save()

                if form.cleaned_data['do_pairs']:
                    docs1 = set([m.document for m in m2mdocs])
                    for motif2 in motifs[motif1ind + 1:]:
                        # if default_score == 'probability':
                        #     m2mdocs2 = DocumentMass2Motif.objects.filter(mass2motif = motif2,probability__gte = doc_m2m_threshold)
                        # else:
                        #     m2mdocs2 = DocumentMass2Motif.objects.filter(mass2motif = motif2,overlap_score__gte = doc_m2m_threshold)
                        m2mdocs2 = get_docm2m(motif2)
                        docs2 = set([m.document for m in m2mdocs2])
                        # Find the intersect
                        intersect = list(docs1 & docs2)
                        if len(intersect) > 0:
                            doc_indices = []
                            for doc in intersect:
                                doc_indices.append(list(documents).index(doc))
                            n_above = 0
                            n_below = 0
                            for ind in doc_indices:
                                if logfc[ind] < lowperc_value:
                                    n_below += 1
                                if logfc[ind] > upperc_value:
                                    n_above += 1
                            score_list = []
                            n_sub_docs = len(intersect)
                            up_range = range(n_above, n_sub_docs + 1)
                            down_range = range(n_below, n_sub_docs + 1)
                            score_list.append(n_sub_docs)
                            score_list.append(n_above)
                            up_score = hypergeom.pmf(up_range, M, total_above, n_sub_docs).sum()
                            score_list.append(up_score)
                            score_list.append(n_below)
                            down_score = hypergeom.pmf(down_range, M, total_below, n_sub_docs).sum()
                            score_list.append(down_score)
                            discrete_scores.append(("{}+{}".format(motif.name, motif2.name), score_list))
                n_done += 1
                print n_done

            context_dict['total_above'] = total_above
            context_dict['total_below'] = total_below
            context_dict['discrete_scores'] = discrete_scores
            context_dict['group1'] = form.cleaned_data['group1']
            context_dict['group2'] = form.cleaned_data['group2']

        else:
            # invalid form
            context_dict['topicscoringform'] = form
    else:
        form = TopicScoringForm(choices)
        context_dict['topicscoringform'] = form

    return render(request, 'basicviz/compute_topic_scores.html', context_dict)

@login_required(login_url='/registration/login/')
def show_docs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You don't have permission to access this page")
    documents = Document.objects.filter(experiment=experiment)
    print len(documents)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['documents'] = documents
    context_dict['n_docs'] = len(documents)
    return render(request, 'basicviz/show_docs.html', context_dict)

@login_required(login_url='/registration/login/')
def show_doc(request, doc_id):
    document = Document.objects.get(id=doc_id)
    experiment = document.experiment

    if not check_user(request,experiment):
        return index(request)
    print document.experiment.experiment_type
    if document.experiment.experiment_type == '0':
        context_dict = get_doc_context_dict(document)
    elif document.experiment.experiment_type == '1':
        context_dict = get_decomp_doc_context_dict(document)
    else:
        context_dict = {}
    print context_dict
    context_dict['document'] = document
    context_dict['experiment'] = experiment

    if document.csid:
        context_dict['csid'] = document.csid
        
    if document.image_url:
        context_dict['image_url'] = document.image_url

    return render(request, 'basicviz/show_doc.html', context_dict)


def get_doc_context_dict(document):
    features = FeatureInstance.objects.filter(document=document)
    context_dict = {}
    context_dict['features'] = features
    experiment = document.experiment
    doc_m2m_threshold = get_option('doc_m2m_threshold', experiment=experiment)
    
    mass2motif_instances = get_docm2m_bydoc(document)
    context_dict['mass2motifs'] = mass2motif_instances
    feature_mass2motif_instances = []
    for feature in features:
        if feature.intensity > 0:
            feature_mass2motif_instances.append(
                (feature, FeatureMass2MotifInstance.objects.filter(featureinstance=feature)))

    feature_mass2motif_instances = sorted(feature_mass2motif_instances, key=lambda x: x[0].intensity, reverse=True)
    context_dict['fm2m'] = feature_mass2motif_instances
    return context_dict

@login_required(login_url='/registration/login/')
def view_parents(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    experiment = motif.experiment
    if not check_user(request,experiment):
        return HttpResponse("You don't have permission to access this page")
    print 'Motif metadata', motif.metadata

    context_dict = {'mass2motif': motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')
    total_prob = sum([m.probability for m in motif_features])
    context_dict['motif_features'] = motif_features
    context_dict['total_prob'] = total_prob

    # Get the taxa or substituent terms (if there are any)
    taxa_terms = motif.taxainstance_set.all().order_by('-probability')
    substituent_terms = motif.substituentinstance_set.all().order_by('-probability')

    if len(taxa_terms) > 0:
        context_dict['taxa_terms'] = taxa_terms
    if len(substituent_terms) > 0:
        context_dict['substituent_terms'] = substituent_terms

    # doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = motif.experiment)
    # if doc_m2m_threshold:
    #     doc_m2m_threshold = float(doc_m2m_threshold)
    # else:
    #     doc_m2m_threshold = 0.00 # Default value

    # default_score = get_option('default_doc_m2m_score',experiment = motif.experiment)
    # if not default_score:
    #     default_score = 'probability'

    # if default_score == 'probability':
    #     dm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold)
    # else:
    #     dm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = doc_m2m_threshold)

    dm2m = get_docm2m(motif)
    context_dict['dm2ms'] = dm2m

    context_dict['status'] = 'Edit metadata...'
    if request.method == 'POST':
        form = Mass2MotifMetadataForm(request.POST)
        if form.is_valid():
            new_annotation = form.cleaned_data['metadata']
            new_short_annotation = form.cleaned_data['short_annotation']
            md = jsonpickle.decode(motif.metadata)
            if len(new_annotation) > 0:
                md['annotation'] = new_annotation
            elif 'annotation' in md:
                del md['annotation']
            if len(new_short_annotation) > 0:
                md['short_annotation'] = new_short_annotation
            elif 'short_annotation' in md:
                del md['short_annotation']
            motif.metadata = jsonpickle.encode(md)
            motif.save()
            context_dict['status'] = 'Metadata saved...'

    permission = check_user(request,experiment)
    if permission == 'edit':
        metadata_form = Mass2MotifMetadataForm(
            initial={'metadata': motif.annotation, 'short_annotation': motif.short_annotation})
        context_dict['metadata_form'] = metadata_form

    massbank_form = get_massbank_form(motif, motif_features)
    context_dict['massbank_form'] = massbank_form

    return render(request, 'basicviz/view_parents.html', context_dict)

@login_required(login_url='/registration/login/')
def mass2motif_feature(request, fm2m_id):
    mass2motif_feature = Mass2MotifInstance.objects.get(id=fm2m_id)
    context_dict = {}
    context_dict['mass2motif_feature'] = mass2motif_feature

    total_intensity = 0.0
    topic_intensity = 0.0
    n_docs = 0
    feature_instances = FeatureInstance.objects.filter(feature=mass2motif_feature.feature)
    docs = []
    for instance in feature_instances:
        total_intensity += instance.intensity
        fi_m2m = FeatureMass2MotifInstance.objects.filter(featureinstance=instance,
                                                          mass2motif=mass2motif_feature.mass2motif)
        if len(fi_m2m) > 0:
            topic_intensity += fi_m2m[0].probability * instance.intensity
            if fi_m2m[0].probability >= 0.75:
                n_docs += 1
                docs.append(instance.document)

    context_dict['total_intensity'] = total_intensity
    context_dict['topic_intensity'] = topic_intensity
    context_dict['intensity_perc'] = 100.0 * topic_intensity / total_intensity
    context_dict['n_docs'] = n_docs
    context_dict['docs'] = docs

    return render(request, 'basicviz/mass2motif_feature.html', context_dict)


def get_parents(request, motif_id, vo_id):
    viz_options = VizOptions.objects.get(id=vo_id)
    experiment = viz_options.experiment
    if experiment.experiment_type == '0': #ms2lda
        motif = Mass2Motif.objects.get(id=motif_id)
        edge_choice = viz_options.edge_choice
        if edge_choice == 'probability':
            docm2m = DocumentMass2Motif.objects.filter(mass2motif=motif, probability__gte=viz_options.edge_thresh).order_by(
                '-probability')
        else:
            docm2m = DocumentMass2Motif.objects.filter(mass2motif=motif,
                                                       overlap_score__gte=viz_options.edge_thresh).order_by(
                '-overlap_score')
        documents = [d.document for d in docm2m]
        parent_data = []
        for dm in docm2m:
            document = dm.document
            if viz_options.just_annotated_docs and document.annotation:
                parent_data.append(get_doc_for_plot(document.id, motif_id,score_type = edge_choice))
            elif not viz_options.just_annotated_docs:
                parent_data.append(get_doc_for_plot(document.id, motif_id,score_type = edge_choice))
    else: # decomposition
        parent_data = get_parents_decomposition(motif_id,vo_id = vo_id,experiment = experiment)
    return HttpResponse(json.dumps(parent_data), content_type='application/json')




def get_parents_no_vo(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)

    # doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = motif.experiment)
    # if doc_m2m_threshold:
    #     doc_m2m_threshold = float(doc_m2m_threshold)
    # else:
    #     doc_m2m_threshold = 0.00 # Default value

    # default_score = get_option('default_doc_m2m_score',experiment = motif.experiment)
    # if not default_score:
    #     default_score = 'probability'

    # if default_score == 'probability':
    #     docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold).order_by('-probability')
    # else:
    #     docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = doc_m2m_threshold).order_by('-overlap_score')
    docm2m = get_docm2m(motif)
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        document = dm.document
        parent_data.append(get_doc_for_plot(document.id, motif_id))
    return HttpResponse(json.dumps(parent_data), content_type='application/json')

# Method to get the metadata for all parent ions in an experiment
# Returns a json object
def get_all_parents_metadata(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    documents = Document.objects.filter(experiment = experiment)
    parent_data = []
    for document in documents:
        parent_data.append(jsonpickle.decode(document.metadata))
    return HttpResponse(json.dumps(parent_data), content_type =  'application/json')


def get_parents_metadata(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    docm2m = get_docm2m(motif)
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        document = dm.document
        parent_data.append(document.metadata)
    return HttpResponse(json.dumps(parent_data), content_type='application/json')


# def get_annotated_parents(request,motif_id):
#     motif = Mass2Motif.objects.get(id=motif_id)
#     docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif)
#     documents = [d.document for d in docm2m]
#     parent_data = []
#     for dm in docm2m:
#         if dm.probability > 0.05:
#             document = dm.document
#             if len(document.annotation) > 0:
#                 parent_data.append(get_doc_for_plot(document.id,motif_id))
#     return HttpResponse(json.dumps(parent_data),content_type = 'application/json')

def get_word_graph(request, motif_id, vo_id, experiment = None):
    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id = vo_id)
        experiment = viz_options.experiment
        edge_thresh = viz_options.edge_thresh
        edge_choice = viz_options.edge_choice
    elif experiment:
        edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
        edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)
    else:
        motif = Mass2Motif.objects.get(id = motif_id)
        experiment = motif.experiment
        edge_choice = 'probability'
        edge_thresh = 0.05


    if experiment.experiment_type == "0": # standard LDA
        motif = Mass2Motif.objects.get(id = motif_id)
        m2mIns = Mass2MotifInstance.objects.filter(mass2motif = motif, probability__gte = 0.01)
        if edge_choice == 'probability':
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh)
        else:
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh)
        data_for_json = []
        data_for_json.append(len(docm2ms))
        feat_counts = {}
        for feature in m2mIns:
            feat_counts[feature.feature] = 0
        for dm2m in docm2ms:
            fi = FeatureInstance.objects.filter(document = dm2m.document)
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
    else:
        data_for_json = []
 
    return HttpResponse(json.dumps(data_for_json), content_type='application/json')


def view_word_graph(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    context_dict = {'mass2motif': motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')


    context_dict['motif_features'] = motif_features
    return render(request, 'basicviz/view_word_graph.html', context_dict)


def get_intensity(request, motif_id, vo_id, experiment = None):
    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id = vo_id)
        experiment = viz_options.experiment
        edge_thresh = viz_options.edge_thresh
        edge_choice = viz_options.edge_choice
    elif experiment:
        edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
        edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)
    else:
        motif = Mass2Motif.objects.get(id = motif_id)
        experiment = motif.experiment
        edge_choice = 'probability'
        edge_thresh = 0.05

    colours = ['#404080', '#0080C0']
    colours = ['red','blue']


    if experiment.experiment_type == "0": # standard LDA
        motif = Mass2Motif.objects.get(id = motif_id)
        m2mIns = Mass2MotifInstance.objects.filter(mass2motif = motif, probability__gte = 0.01)
        if edge_choice == 'probability':
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh)
        else:
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh)
        documents = [d.document for d in docm2ms]
        data_for_json = []
        feat_total_intensity = {}
        feat_motif_intensity = {}
        features = [m.feature for m in m2mIns]
        for feature in features:
            feat_total_intensity[feature] = 0.0
            feat_motif_intensity[feature] = 0.0
        for feature in features:
            fi = FeatureInstance.objects.filter(feature = feature)
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
    else:
        data_for_json = []

    return HttpResponse(json.dumps(data_for_json), content_type='application/json')

@login_required(login_url='/registration/login/')
def view_intensity(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)

    experiment = motif.experiment

    context_dict = {'mass2motif': motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')

    context_dict['motif_features'] = motif_features
    return render(request, 'basicviz/view_intensity.html', context_dict)

@login_required(login_url='/registration/login/')
def view_mass2motifs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    if experiment.experiment_type == '0':
        mass2motifs = Mass2Motif.objects.filter(experiment=experiment).order_by('name')
        context_dict = {'mass2motifs': mass2motifs}
        context_dict['experiment'] = experiment
        return render(request, 'basicviz/view_mass2motifs.html', context_dict)
    elif experiment.experiment_type == '1': #decomp
        raise Http404('Page not found')
        # documents = Document.objects.filter(experiment = experiment)
        # dm2m = DocumentGlobalMass2Motif.objects.filter(document__in = documents)
        # mass2motifs = list(set([d.mass2motif for d in dm2m]))
        # context_dict = {'mass2motifs':mass2motifs,'experiment':experiment}
        return render(request, 'decomposition/view_mass2motifs.html',context_dict)



def get_doc_for_plot(doc_id, motif_id=None, get_key=False,score_type = None):
    colours = ['red', 'green', 'black', 'yellow']
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document=document)
    plot_fragments = []

    # Get the parent info
    metadata = jsonpickle.decode(document.metadata)
    if 'parentmass' in metadata:
        parent_mass = float(metadata['parentmass'])
    elif 'mz' in metadata:
        parent_mass = float(metadata['mz'])
    elif '_' in document.name:
        try:
            parent_mass = float(document.name.split('_')[0])
        except:
            # in case the name isnt that format
            parent_mass = 0.0
    else:
        parent_mass = 0.0
    probability = "na"

    default_score = get_option('default_doc_m2m_score', experiment=document.experiment)
    if not default_score:
        default_score = 'probability'

    if not motif_id == None:
        m2m = Mass2Motif.objects.get(id=motif_id)
        dm2m = DocumentMass2Motif.objects.get(mass2motif=m2m, document=document)
        if not score_type:
            if default_score == 'probability':
                probability = dm2m.probability
            else:
                probability = dm2m.overlap_score
        else:
            if score_type == 'probability':
                probability = dm2m.probability
            else:
                probability = dm2m.overlap_score

    parent_data = (parent_mass, 100.0, document.display_name, document.annotation, probability)
    plot_fragments.append(parent_data)
    child_data = []

    # Only colours the first five
    if motif_id == None:
        topic_colours = {}
        if default_score == 'probability':
            topics = sorted(DocumentMass2Motif.objects.filter(document=document), key=lambda x: x.probability,
                            reverse=True)
        else:
            topics = sorted(DocumentMass2Motif.objects.filter(document=document), key=lambda x: x.overlap_score,
                            reverse=True)
        topics_to_plot = []
        for i in range(4):
            if i == len(topics):
                break
            topics_to_plot.append(topics[i].mass2motif)
            topic_colours[topics[i].mass2motif] = colours[i]
    else:
        topic = Mass2Motif.objects.get(id=motif_id)
        topics_to_plot = [topic]
        topic_colours = {topic: 'red'}

    max_intensity = 0.0
    for feature_instance in features:
        if feature_instance.intensity > max_intensity:
            max_intensity = feature_instance.intensity

    if len(features) > 0:
        for feature_instance in features:
            phi_values = FeatureMass2MotifInstance.objects.filter(featureinstance=feature_instance)
            mass = float(feature_instance.feature.name.split('_')[1])
            this_intensity = feature_instance.intensity * 100.0 / max_intensity
            feature_name = feature_instance.feature.name
            if feature_name.startswith('fragment'):
                cum_pos = 0.0
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = phi_value.probability * this_intensity
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append((mass, mass, cum_pos, cum_pos + proportion, 1, colour, feature_name))
                        cum_pos += proportion
                    else:
                        proportion = phi_value.probability * this_intensity
                        other_topics += proportion
                child_data.append((mass, mass, this_intensity - other_topics, this_intensity, 1, 'gray', feature_name))
            else:
                cum_pos = parent_mass - mass
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = mass * phi_value.probability
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append(
                            (cum_pos, cum_pos + proportion, this_intensity, this_intensity, 0, colour, feature_name))
                        cum_pos += proportion
                    else:
                        proportion = mass * phi_value.probability
                        other_topics += proportion
                child_data.append(
                    (parent_mass - other_topics, parent_mass, this_intensity, this_intensity, 0, 'gray', feature_name))
    plot_fragments.append(child_data)

    if get_key:
        key = []
        for topic in topic_colours:
            key.append((topic.name, topic_colours[topic]))
        return [plot_fragments], key

    return plot_fragments


def get_doc_topics(request, doc_id):
    document = Document.objects.get(id = doc_id)
    if document.experiment.experiment_type == '0':
        plot_fragments = [get_doc_for_plot(doc_id, get_key=True)]
    elif document.experiment.experiment_type == '1': # decomposition
        raise Http404('Page not found')
        # score_type = get_option('default_doc_m2m_score',experiment = document.experiment)
        # if not score_type:
        #     score_type = 'probability'
        # plot_fragments = [get_parent_for_plot_decomp(document,edge_choice=score_type,get_key = True)]
    else:
        plot_fragments = []
    return HttpResponse(json.dumps(plot_fragments), content_type='application/json')

@login_required(login_url = '/registration/login/')
def start_viz(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {'experiment': experiment}

    if request.method == 'POST':
        viz_form = VizForm(request.POST)
        if viz_form.is_valid():
            min_degree = viz_form.cleaned_data['min_degree']
            edge_thresh = viz_form.cleaned_data['edge_thresh']
            j_a_n = viz_form.cleaned_data['just_annotated_docs']
            colour_by_logfc = viz_form.cleaned_data['colour_by_logfc']
            discrete_colour = viz_form.cleaned_data['discrete_colour']
            lower_colour_perc = viz_form.cleaned_data['lower_colour_perc']
            upper_colour_perc = viz_form.cleaned_data['upper_colour_perc']
            colour_topic_by_score = viz_form.cleaned_data['colour_topic_by_score']
            random_seed = viz_form.cleaned_data['random_seed']
            edge_choice = viz_form.cleaned_data['edge_choice']
            edge_choice = edge_choice[0].encode('ascii', 'ignore')  # should turn the unicode into ascii
            vo = VizOptions.objects.get_or_create(experiment=experiment,
                                                  min_degree=min_degree,
                                                  edge_thresh=edge_thresh,
                                                  just_annotated_docs=j_a_n,
                                                  colour_by_logfc=colour_by_logfc,
                                                  discrete_colour=discrete_colour,
                                                  lower_colour_perc=lower_colour_perc,
                                                  upper_colour_perc=upper_colour_perc,
                                                  colour_topic_by_score=colour_topic_by_score,
                                                  random_seed=random_seed,
                                                  edge_choice=edge_choice)[0]
            context_dict['viz_options'] = vo

        else:
            context_dict['viz_form'] = viz_form
    else:
        viz_form = VizForm()
        context_dict['viz_form'] = viz_form

    if 'viz_form' in context_dict:
        return render(request, 'basicviz/viz_form.html', context_dict)
    else:
        # initial_motif = Mass2Motif.objects.filter(experiment=experiment)[0]
        # context_dict['initial_motif'] = initial_motif
        return render(request, 'basicviz/graph.html', context_dict)

@login_required(login_url = '/registration/login/')
def start_annotated_viz(request, experiment_id):
    # Is this function ever called??
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")

    context_dict = {'experiment': experiment}
    # This is a bit of a hack to make sure that the initial motif is one in the graph
    documents = Document.objects.filter(experiment=experiment)
    for document in documents:
        if len(document.annotation) > 0:
            for docm2m in DocumentMass2Motif.objects.filter(document=document):
                if docm2m.probability > 0.05:
                    context_dict['initial_motif'] = docm2m.mass2motif
                    break
    return render(request, 'basicviz/annotated_graph.html', context_dict)


def get_graph(request, vo_id):
    viz_options = VizOptions.objects.get(id=vo_id)
    experiment = viz_options.experiment

    if experiment.experiment_type == "0":
        G = make_graph(experiment, min_degree=viz_options.min_degree,
                       edge_thresh=viz_options.edge_thresh,
                       just_annotated_docs=viz_options.just_annotated_docs,
                       colour_by_logfc=viz_options.colour_by_logfc,
                       discrete_colour=viz_options.discrete_colour,
                       lower_colour_perc=viz_options.lower_colour_perc,
                       upper_colour_perc=viz_options.upper_colour_perc,
                       colour_topic_by_score=viz_options.colour_topic_by_score,
                       edge_choice=viz_options.edge_choice)
    else:
        # G = make_decomposition_graph(experiment, min_degree=viz_options.min_degree,
        #                edge_thresh=viz_options.edge_thresh,
        #                edge_choice=viz_options.edge_choice)
        raise Http404("page not found")
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d), content_type='application/json')


# def make_decomposition_graph(experiment,min_degree = 5,edge_thresh = 0.5,
#                                 edge_choice = 'probability',topic_scale_factor = 5, edge_scale_factor = 5):
#     # This is the graph maker for a decomposition experiment
#     documents = Document.objects.filter(experiment = experiment)
#     doc_motif = DocumentGlobalMass2Motif.objects.filter(document__in = documents)
#     G = nx.Graph()
#     motif_degrees = {}
#     for dm in doc_motif:
#         if edge_choice == 'probability':
#             edge_score = dm.probability
#         else:
#             edge_score = dm.overlap_score
#         if edge_score >= edge_thresh:
#             if not dm.mass2motif in motif_degrees:
#                 motif_degrees[dm.mass2motif] = 1
#             else:
#                 motif_degrees[dm.mass2motif] += 1
#     used_motifs = []
#     for motif,degree in motif_degrees.items():
#         if degree >= min_degree:
#             # add to the graph
#             used_motifs.append(motif)
#             metadata = jsonpickle.decode(motif.originalmotif.metadata)
#             if 'annotation' in metadata:
#                 G.add_node(motif.originalmotif.name, group=2, name=metadata['annotation'],
#                            size=topic_scale_factor * degree,
#                            special=True, in_degree = degree,
#                            score=1, node_id=motif.id, is_topic=True)
#             else:
#                 G.add_node(motif.originalmotif.name, group=2, name=motif.originalmotif.name,
#                            size=topic_scale_factor * degree,
#                            special=False, in_degree=degree,
#                            score=1, node_id=motif.id, is_topic=True)
#     used_docs = []
#     for dm in doc_motif:
#         if dm.mass2motif in used_motifs:
#             # add the edge
#             if not dm.document in used_docs:
#                 # add the document node
#                 metadata = jsonpickle.decode(dm.document.metadata)
#                 if 'compound' in metadata:
#                     name = metadata['compound']
#                 elif 'annotation' in metadata:
#                     name = metadata['annotation']
#                 else:
#                     name = dm.document.name

#                 G.add_node(dm.document.name, group=1, name=name, size=20,
#                            type='square', peakid=dm.document.name, special=False,
#                            in_degree=0, score=0, is_topic=False)
#                 used_docs.append(dm.document)
#             if edge_choice == 'probability':
#                 weight = edge_scale_factor * dm.probability
#             else:
#                 weight = edge_scale_factor * dm.overlap_score
#             G.add_edge(dm.mass2motif.originalmotif.name, dm.document.name, weight=weight)


#     return G


def make_graph(experiment, edge_thresh=0.05, min_degree=5,
               topic_scale_factor=5, edge_scale_factor=5, just_annotated_docs=False,
               colour_by_logfc=False, discrete_colour=False, lower_colour_perc=10, upper_colour_perc=90,
               colour_topic_by_score=False, edge_choice='probability'):
    mass2motifs = Mass2Motif.objects.filter(experiment=experiment)
    # Find the degrees
    topics = {}
    for mass2motif in mass2motifs:
        topics[mass2motif] = 0
        if edge_choice == 'probability':
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=edge_thresh)
        else:
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, overlap_score__gte=edge_thresh)


        for d in docm2ms:
            if just_annotated_docs and d.document.annotation:
                topics[mass2motif] += 1
            elif (not just_annotated_docs):
                topics[mass2motif] += 1
    to_remove = []
    for topic in topics:
        if topics[topic] < min_degree:
            to_remove.append(topic)
    for topic in to_remove:
        del topics[topic]

    print "First"
    # Add the topics to the graph
    G = nx.Graph()
    for topic in topics:
        metadata = jsonpickle.decode(topic.metadata)
        if colour_topic_by_score:
            upscore = metadata.get('upscore', 1.0)
            downscore = metadata.get('downscore', 1.0)
            if upscore < 0.05:
                highlight_colour = '#0000FF'
            elif downscore < 0.05:
                highlight_colour = '#FF0000'
            else:
                highlight_colour = '#AAAAAA'
            name = metadata.get('annotation', topic.name)
            G.add_node(topic.name, group=2, name=name,
                       size=topic_scale_factor * topics[topic],
                       special=True, in_degree=topics[topic],
                       score=1, node_id=topic.id, is_topic=True,
                       highlight_colour=highlight_colour)

        else:
            if topic.annotation:
            # if 'annotation' in metadata:
                G.add_node(topic.name, group=2, name=topic.annotation,
                           size=topic_scale_factor * topics[topic],
                           special=True, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)
            else:
                G.add_node(topic.name, group=2, name=topic.name,
                           size=topic_scale_factor * topics[topic],
                           special=False, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)

    documents = Document.objects.filter(experiment=experiment)
    if colour_by_logfc:
        all_logfc_vals = []
        if colour_by_logfc:
            for document in documents:
                if document.logfc:
                    val = float(document.logfc)
                    if not np.abs(val) == np.inf:
                        all_logfc_vals.append(float(document.logfc))
        logfc_vals = np.sort(np.array(all_logfc_vals))

        perc_lower = logfc_vals[int(np.floor((lower_colour_perc / 100.0) * len(logfc_vals)))]
        perc_upper = logfc_vals[int(np.ceil((upper_colour_perc / 100.0) * len(logfc_vals)))]

        lowcol = [255, 0, 0]
        endcol = [0, 0, 255]

    if just_annotated_docs:
        new_documents = []
        for document in documents:
            if document.annotation:
                new_documents.append(document)

        documents = new_documents

    doc_nodes = []

    print "Second"

    if edge_choice == 'probability':
        docm2mset = DocumentMass2Motif.objects.filter(document__in=documents, mass2motif__in=topics,
                                                      probability__gte=edge_thresh)
    else:
        docm2mset = DocumentMass2Motif.objects.filter(document__in=documents, mass2motif__in=topics,
                                                      overlap_score__gte=edge_thresh)

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
            if not colour_by_logfc:
                G.add_node(docm2m.document.name, group=1, name=name, size=20,
                           type='square', peakid=docm2m.document.name, special=False,
                           in_degree=0, score=0, is_topic=False)
            else:
                if docm2m.document.logfc:
                    lfc = float(docm2m.document.logfc)
                    if lfc > perc_upper or lfc == np.inf:
                        col = "#{}{}{}".format('00', '00', 'FF')
                    elif lfc < perc_lower or -lfc == np.inf:
                        col = "#{}{}{}".format('FF', '00', '00')
                    else:
                        if not discrete_colour:
                            pos = (lfc - perc_lower) / (perc_upper - perc_lower)
                            r = lowcol[0] + int(pos * (endcol[0] - lowcol[0]))
                            g = lowcol[1] + int(pos * (endcol[1] - lowcol[1]))
                            b = lowcol[2] + int(pos * (endcol[2] - lowcol[2]))
                            col = "#{}{}{}".format("{:02x}".format(r), "{:02x}".format(g), "{:02x}".format(b))
                        else:
                            col = '#FFFFFF'
                else:
                    col = '#FFFFFF'
                G.add_node(docm2m.document.name, group=1, name=name, size=20,
                           type='square', peakid=docm2m.document.name, special=True,
                           highlight_colour=col, logfc=docm2m.document.logfc,
                           in_degree=0, score=0, is_topic=False)

            doc_nodes.append(docm2m.document)

        if edge_choice == 'probability':
            weight = edge_scale_factor * docm2m.probability
        else:
            weight = edge_scale_factor * docm2m.overlap_score
        G.add_edge(docm2m.mass2motif.name, docm2m.document.name, weight=weight)
    print "Third"
    return G

@login_required(login_url='/registration/login/')
def topic_pca(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {'experiment': experiment}
    url = '/basicviz/get_topic_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request, 'basicviz/pca.html', context_dict)

@login_required(login_url='/registration/login/')
def document_pca(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {}
    context_dict['experiment'] = experiment
    url = '/basicviz/get_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request, 'basicviz/pca.html', context_dict)


def get_topic_pca_data(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)
    features = Feature.objects.filter(experiment=experiment)

    n_motifs = len(motifs)
    n_features = len(features)

    mat = []
    motif_index = {}
    motif_pos = 0

    for motif in motifs:
        motif_index[motif] = motif_pos
        motif_pos += 1

    feature_pos = 0
    feature_index = {}

    for feature in features:
        instances = Mass2MotifInstance.objects.filter(feature=feature)
        if len(instances) > 2:  # minimum to include
            feature_index[feature] = feature_pos
            new_row = [0.0 for i in range(n_motifs)]
            for instance in instances:
                motif_pos = motif_index[instance.mass2motif]
                new_row[motif_pos] = instance.probability
            feature_pos += 1
            mat.append(new_row)

    mat = np.array(mat).T

    pca = PCA(n_components=2, whiten=True)
    pca.fit(mat)

    X = pca.transform(mat)
    pca_points = []
    for motif in motif_index:
        motif_pos = motif_index[motif]
        new_element = (X[motif_pos, 0], X[motif_pos, 1], motif.name, '#FF66CC')
        pca_points.append(new_element)

    max_x = np.abs(X[:, 0]).max()
    max_y = np.abs(X[:, 1]).max()

    factors = pca.components_
    max_factor_x = np.abs(factors[0, :]).max()
    factors[0, :] *= max_x / max_factor_x
    max_factor_y = np.abs(factors[1, :]).max()
    factors[1, :] *= max_y / max_factor_y

    pca_lines = []
    factor_colour = 'rgba(0,0,128,0.5)'
    for feature in feature_index:
        xval = factors[0, feature_index[feature]]
        yval = factors[1, feature_index[feature]]
        if abs(xval) > 0.01 * max_x or abs(yval) > 0.01 * max_y:
            pca_lines.append((xval, yval, feature.name, factor_colour))
    # add the weightings
    pca_data = (pca_points, pca_lines)

    return HttpResponse(json.dumps(pca_data), content_type='application/json')


def get_pca_data(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    theta_data = []
    documents = Document.objects.filter(experiment=experiment)
    mass2motifs = Mass2Motif.objects.filter(experiment=experiment)
    n_mass2motifs = len(mass2motifs)
    m2mindex = {}
    msmpos = 0

    # doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = experiment)
    # if doc_m2m_threshold:
    #     doc_m2m_threshold = float(doc_m2m_threshold)
    # else:
    #     doc_m2m_threshold = 0.00 # Default value



    for document in documents:
        new_theta = [0 for i in range(n_mass2motifs)]
        # dm2ms = DocumentMass2Motif.objects.filter(document = document,probability__gte = doc_m2m_threshold)
        dm2ms = get_docm2m_bydoc(document)
        for dm2m in dm2ms:
            if dm2m.mass2motif.name in m2mindex:
                m2mpos = m2mindex[dm2m.mass2motif.name]
            else:
                m2mpos = msmpos
                m2mindex[dm2m.mass2motif.name] = m2mpos
                msmpos += 1
            new_theta[m2mpos] = dm2m.probability
        theta_data.append(new_theta)

    pca = PCA(n_components=2, whiten=True)
    pca.fit(theta_data)

    pca_data = []
    X = pca.transform(theta_data)
    for i in range(len(documents)):
        name = documents[i].name
        md = jsonpickle.decode(documents[i].metadata)
        color = '#ff3333'
        if 'annotation' in md:
            name = md['annotation']
            color = '#BE84CF'
        new_value = (float(X[i, 0]), float(X[i, 1]), name, color)
        pca_data.append(new_value)

    # pca_data = []
    return HttpResponse(json.dumps(pca_data), content_type='application/json')

@login_required(login_url='/registration/login/')
def validation(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {}
    if request.method == 'POST':
        form = ValidationForm(request.POST)
        if form.is_valid():
            p_thresh = form.cleaned_data['p_thresh']
            just_annotated = form.cleaned_data['just_annotated']
            mass2motifs = Mass2Motif.objects.filter(experiment=experiment)
            annotated_mass2motifs = []

            # default_score = get_option('default_doc_m2m_score',experiment = experiment)
            # if not default_score:
            #     default_score = 'probability'

            counts = []
            all_dm2ms = []
            for mass2motif in mass2motifs:
                if mass2motif.annotation:
                    annotated_mass2motifs.append(mass2motif)
                    # if default_score == 'probability':
                    #     dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = p_thresh)
                    # else:
                    #     dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,ovelap_score__gte = p_thresh)
                    dm2ms = get_docm2m(mass2motif, doc_m2m_threshold=p_thresh)
                    tot = 0
                    val = 0
                    for d in dm2ms:
                        if (just_annotated and d.document.annotation) or not just_annotated:
                            tot += 1
                            if d.validated:
                                val += 1
                    counts.append((tot, val))
                    all_dm2ms.append(dm2ms)
                    print dm2ms
            annotated_mass2motifs = zip(annotated_mass2motifs, counts, all_dm2ms)
            context_dict['annotated_mass2motifs'] = annotated_mass2motifs
            context_dict['counts'] = counts
            context_dict['p_thresh'] = p_thresh
            context_dict['just_annotated'] = just_annotated

        else:
            context_dict['validation_form'] = form
    else:

        form = ValidationForm()
        context_dict['validation_form'] = form
    context_dict['experiment'] = experiment
    return render(request, 'basicviz/validation.html', context_dict)

def toggle_dm2m(request, experiment_id, dm2m_id):
    permission = check_user(request,experiment)
    dm2m = DocumentMass2Motif.objects.get(id=dm2m_id)
    jd = []
    if permission == 'edit':
        if dm2m.validated:
            dm2m.validated = False
            jd.append('No')
        else:
            dm2m.validated = True
            jd.append('Yes')
        dm2m.save()
    else:
        if dm2m.validated:
            jd.append('Yes')
        else:
            jd.append('No')
    return HttpResponse(json.dumps(jd), content_type='application/json')
    # return validation(request,experiment_id)


def dump_validations(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment=experiment)
    annotated_mass2motifs = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="valid_dump_{}.csv"'.format(experiment_id)
    writer = csv.writer(response)
    writer.writerow(['msm_id', 'm2m_name', 'm2m_annotation', 'doc_id', 'doc_annotation', 'valid', 'probability'])

    for mass2motif in annotated_mass2motifs:
        dm2ms = get_docm2m(mass2motif)
        for dm2m in dm2ms:
            document = dm2m.document
            # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
            doc_name = '"' + dm2m.document.display_name + '"'
            annotation = '"' + mass2motif.annotation + '"'
            writer.writerow([mass2motif.id, mass2motif.name, mass2motif.annotation.encode('utf8'), dm2m.document.id,
                             doc_name.encode('utf8'), dm2m.validated, dm2m.probability])

    # return HttpResponse(outstring,content_type='text')
    return response


def dump_topic_molecules(request, m2m_id):
    mass2motif = Mass2Motif.objects.get(id=m2m_id)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="topic_molecules_{}.csv"'.format(mass2motif.id)
    writer = csv.writer(response)
    writer.writerow(
        ['m2m_id', 'm2m_name', 'm2m_annotation', 'doc_id', 'doc_annotation', 'valid', 'probability', 'overlap_score', 'doc_csid',
         'doc_inchi'])

    dm2ms = get_docm2m(mass2motif)
    for dm2m in dm2ms:
        document = dm2m.document
        # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
        doc_name = '"' + dm2m.document.display_name + '"'
        annotation = '"' + mass2motif.annotation + '"'
        writer.writerow([mass2motif.id, mass2motif.name, mass2motif.annotation.encode('utf8'), dm2m.document.id,
                         doc_name.encode('utf8'), dm2m.validated, dm2m.probability, dm2m.overlap_score, dm2m.document.csid,
                         dm2m.document.inchikey])

    return response


def get_docm2m(mass2motif, default_score=None, doc_m2m_threshold=None):
    experiment = mass2motif.experiment
    if not default_score:
        default_score = get_option('default_doc_m2m_score', experiment=experiment)
        if not default_score:
            default_score = 'probability'
    if not doc_m2m_threshold:
        doc_m2m_threshold = get_option('doc_m2m_threshold', experiment=experiment)
        if doc_m2m_threshold:
            doc_m2m_threshold = float(doc_m2m_threshold)
        else:
            doc_m2m_threshold = 0.0  # Default

    if default_score == 'probability':
        dm2m = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=doc_m2m_threshold).order_by(
            '-probability')
    elif default_score == 'overlap_score':
        dm2m = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, overlap_score__gte=doc_m2m_threshold).order_by(
            '-overlap_score')
    elif default_score == 'both': 
        dm2m = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=doc_m2m_threshold,
                                                 overlap_score__gte=doc_m2m_threshold).order_by('-probability')
    else:
        dm2m = []


    return dm2m


def get_docm2m_bydoc(document, default_score=None, doc_m2m_threshold=None):
    experiment = document.experiment
    if not default_score:
        default_score = get_option('default_doc_m2m_score', experiment=experiment)
        if not default_score:
            default_score = 'probability'
    if not doc_m2m_threshold:
        doc_m2m_threshold = get_option('doc_m2m_threshold', experiment=experiment)
        if doc_m2m_threshold:
            doc_m2m_threshold = float(doc_m2m_threshold)
        else:
            doc_m2m_threshold = 0.0  # Default

    if default_score == 'probability':
        dm2m = DocumentMass2Motif.objects.filter(document=document, probability__gte=doc_m2m_threshold).order_by(
            '-probability')
    elif default_score == 'overlap_score':
        dm2m = DocumentMass2Motif.objects.filter(document=document, overlap_score__gte=doc_m2m_threshold).order_by(
            '-overlap_score')
    elif default_score == 'both':
        dm2m = DocumentMass2Motif.objects.filter(document=document, probability__gte=doc_m2m_threshold,
                                                 overlap_score__gte=doc_m2m_threshold).order_by('-probability')
    else:
        dm2m = []
    return dm2m

@login_required(login_url='/registration/login/')
def extract_docs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to view this page")
    context_dict = {}
    if request.method == 'POST':
        # form has come, get the documents
        form = DocFilterForm(request.POST)
        if form.is_valid():
            all_docs = Document.objects.filter(experiment=experiment)
            selected_docs = {}
            all_m2m = Mass2Motif.objects.filter(experiment=experiment)
            annotated_m2m = []
            for m2m in all_m2m:
                if m2m.annotation:
                    annotated_m2m.append(m2m)
            for doc in all_docs:
                if form.cleaned_data['annotated_only'] and not doc.display_name:
                    # don't keep
                    continue
                else:
                    dm2m = DocumentMass2Motif.objects.filter(document=doc)
                    m2ms = [d.mass2motif for d in dm2m if d.probability > form.cleaned_data['topic_threshold']]
                    if len(list((set(m2ms) & set(annotated_m2m)))) < form.cleaned_data['min_annotated_topics']:
                        # don't keep
                        continue
                    else:
                        selected_docs[doc] = []
                        for d in dm2m:
                            if d.probability > form.cleaned_data['topic_threshold']:
                                if not form.cleaned_data['only_show_annotated']:
                                    selected_docs[doc].append(d)
                                else:
                                    if d.mass2motif.annotation:
                                        selected_docs[doc].append(d)

                        selected_docs[doc] = sorted(selected_docs[doc], key=lambda x: x.probability, reverse=True)
                context_dict['n_docs'] = len(selected_docs)
            context_dict['docs'] = selected_docs
        else:
            context_dict['doc_form'] = form
    else:
        doc_form = DocFilterForm()
        context_dict['doc_form'] = doc_form
    context_dict['experiment'] = experiment
    return render(request, 'basicviz/extract_docs.html', context_dict)


def compute_overlap_score(mass2motif, document):
    # Computes the 'simon' score that looks at the proportion of the
    # mass2motif that is represented in the document
    document_feature_instances = FeatureInstance.objects.filter(document=document)
    # Following are the phi scores
    feature_mass2motif_instances = FeatureMass2MotifInstance.objects.filter(
        featureinstance__in=document_feature_instances)
    score = 0.0
    for feature_mass2motif_instance in feature_mass2motif_instances:
        feature = feature_mass2motif_instance.featureinstance.feature
        m2m_feature = Mass2MotifInstance.objects.filter(mass2motif=mass2motif, feature=feature)
        if len(m2m_feature) > 0:
            score += feature_mass2motif_instance.probability * m2m_feature[0].probability
    return score

@login_required(login_url='/registration/login/')
def rate_by_conserved_motif_rating(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request,experiment):
        return HttpResponse("You do not have permission to access this page")
    mass2motifs = experiment.mass2motif_set.all()
    motif_scores = []

    

    for motif in mass2motifs:
        # motif_docs = motif.documentmass2motif_set.all()
        motif_docs = get_docm2m(motif)
        docs = [m.document for m in motif_docs]
        total_docs = len(motif_docs)
        thresh = total_docs * 0.4
        motif_features = motif.mass2motifinstance_set.all()
        n_matching = 0
        for motif_feature in motif_features:
            # How many docs is it in
            n_docs = len(FeatureInstance.objects.filter(feature=motif_feature.feature, document__in=docs))
            if n_docs > thresh:
                n_matching += 1
        motif_scores.append((motif, n_matching, len(docs)))

    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['motif_scores'] = motif_scores

    return render(request, 'basicviz/rate_by_conserved_motif.html', context_dict)

@login_required(login_url='/registration/login/')
def high_classyfire(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    motifs = Mass2Motif.objects.filter(experiment = experiment)
    taxa_instances = TaxaInstance.objects.filter(motif__in = motifs,probability__gte = 0.2)
    substituent_instances = SubstituentInstance.objects.filter(motif__in = motifs,probability__gte = 0.2)
    context_dict = {}
    context_dict['taxa_instances'] = taxa_instances
    context_dict['substituent_instances'] = substituent_instances
    context_dict['experiment'] = experiment
    return render(request,'basicviz/high_classyfire.html',context_dict)

def get_features(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    features = Feature.objects.filter(experiment = experiment)
    output_features = [(f.name,f.min_mz,f.max_mz) for f in features]
    return HttpResponse(json.dumps(output_features), content_type='application/json')

def get_annotated_topics(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    motifs = Mass2Motif.objects.filter(experiment = experiment)
    output_motifs = []
    for motif in motifs:
        if motif.annotation:
            output_motifs.append(motif)

    output_metadata = []
    output_beta = []
    for motif in output_motifs:
        output_metadata.append((motif.name,motif.annotation))
        betas = []
        beta_vals = motif.mass2motifinstance_set.all()
        for b in beta_vals:
            betas.append((b.feature.name,b.probability))
        output_beta.append((motif.name,betas))

    output = (output_metadata,output_beta)


    return HttpResponse(json.dumps(output),content_type = 'application/json')

# Gets the document <-> m2m links for a particular experiment as a json object
def get_doc_m2m(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    dm2m = DocumentMass2Motif.objects.filter(document__experiment = experiment)
    output_data = []
    for d in dm2m:
        output_data.append([d.mass2motif.name,d.document.name,d.probability,d.overlap_score])
    return HttpResponse(json.dumps(output_data),content_type = 'application/json')

def get_beta(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mi = Mass2MotifInstance.objects.filter(mass2motif__experiment = experiment)
    output_data = []
    for m in mi:
        output_data.append([m.mass2motif.name,m.feature.name,m.probability])
    return HttpResponse(json.dumps(output_data),content_type = 'application/json')


def get_all_doc_data(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    documents = Document.objects.filter(experiment = experiment)
    out_data = []
    for document in documents:
        doc_feat = FeatureInstance.objects.filter(document = document)
        doc_features = [(d.feature.name,d.intensity) for d in doc_feat]
        doc_motif = DocumentMass2Motif.objects.filter(document = document)
        doc_motifs = [(d.mass2motif.name,d.probability,d.overlap_score) for d in doc_motif]
        out_data.append([document.name,doc_features,doc_motifs])
    return HttpResponse(json.dumps(out_data),content_type = 'application/json')

def get_proportion_annotated_docs(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    output_data = []
    documents = Document.objects.filter(experiment = experiment)
    n_docs = len(documents)
    n_annotated = 0
    for document in documents:
        dm2ms = get_docm2m_bydoc(document)
        for dm in dm2ms:
            if dm.mass2motif.annotation:
                n_annotated += 1
                break
    output_data.append((experiment.name,n_docs,n_annotated))
    return HttpResponse(json.dumps(output_data),content_type = 'application/json')

# Renders a page summarising a particular experiment
def summary(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    user_experiments = UserExperiment.objects.filter(experiment = experiment)

    motifs = Mass2Motif.objects.filter(experiment = experiment)
    motif_tuples = []
    for motif in motifs:
        dm2ms = get_docm2m(motif)
        motif_tuples.append((motif,len(dm2ms)))

    motif_features = Mass2MotifInstance.objects.filter(mass2motif__experiment = experiment,probability__gte = 0.05)


    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['user_experiments'] = user_experiments
    context_dict['motif_tuples'] = motif_tuples
    context_dict['motif_features'] = motif_features

    return render(request,'basicviz/summary.html',context_dict)

# Matches motifs in one experiment with those in another
# TODO: move to celery, create form
def start_match_motifs(request,experiment_id):
    context_dict = {}
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict['experiment'] = experiment
    if request.method == 'POST':
        match_motif_form = MatchMotifForm(request.POST)
        if match_motif_form.is_valid():
            base_experiment_id = int(match_motif_form.cleaned_data['other_experiment'])
            minimum_score_to_save = float(match_motif_form.cleaned_data['min_score_to_save'])
            match_motifs.delay(experiment.id,base_experiment_id,min_score_to_save = minimum_score_to_save)
            return redirect('/basicviz/')
    else:
        match_motif_form = MatchMotifForm()
    context_dict['match_motif_form'] = match_motif_form
    return render(request,'basicviz/start_match_motifs.html',context_dict)

def manage_motif_matches(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    matches = MotifMatch.objects.filter(frommotif__experiment = experiment)
    context_dict = {}
    context_dict['matches'] = matches
    context_dict['experiment'] = experiment
    return render(request,'basicviz/match_motifs.html',context_dict)

def add_link(request,from_motif_id,to_motif_id):
    from_motif = Mass2Motif.objects.get(id = from_motif_id)
    to_motif = Mass2Motif.objects.get(id = to_motif_id)
    from_motif.linkmotif = to_motif
    from_motif.save()
    experiment_id = from_motif.experiment.id
    return manage_motif_matches(request,experiment_id)

def remove_link(request,from_motif_id):
    from_motif = Mass2Motif.objects.get(id = from_motif_id)
    from_motif.linkmotif = None
    from_motif.save()
    experiment_id =from_motif.experiment.id
    return manage_motif_matches(request,experiment_id)
