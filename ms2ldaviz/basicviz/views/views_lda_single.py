import csv
import json
from collections import Counter, defaultdict

import jsonpickle
import networkx as nx
import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.http import HttpResponse, Http404
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA

from annotation.models import TaxaInstance, SubstituentInstance
from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.forms import DocFilterForm, ValidationForm, VizForm, \
    TopicScoringForm, MatchMotifForm
from basicviz.models import Feature, Experiment, Document, FeatureInstance, DocumentMass2Motif, \
    FeatureMass2MotifInstance, Mass2Motif, Mass2MotifInstance, VizOptions, UserExperiment, MotifMatch, \
    PublicExperiments, FeatureInstance2Sub
from basicviz.tasks import match_motifs_set
from basicviz.views import index as basicviz_index
from basicviz.views.views_lda_admin import list_all_experiments
from decomposition.decomposition_functions import get_parents_decomposition, get_decomp_doc_context_dict
from massbank.forms import Mass2MotifMetadataForm
from massbank.views import get_massbank_form
from motifdb.models import MDBMotif
from ms1analysis.models import Analysis, AnalysisResult, AnalysisResultPlage
from options.views import get_option


def check_user(request, experiment):
    user = request.user
    if user.is_staff:  # staff can view all experiments
        return 'edit'

    try:
        ue = UserExperiment.objects.get(experiment=experiment, user=user)
        return ue.permission
    except:
        # try the public experiments
        e = PublicExperiments.objects.filter(experiment=experiment)
        if len(e) > 0:
            return "view"
        else:
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
                print(n_done)

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


# @login_required(login_url='/registration/login/')
def show_docs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
        return HttpResponse("You don't have permission to access this page")
    documents = Document.objects.filter(experiment=experiment)
    print(len(documents))
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['documents'] = documents
    context_dict['n_docs'] = len(documents)
    context_dict['is_public'] = is_public(experiment)
    return render(request, 'basicviz/show_docs.html', context_dict)


def is_public(experiment):
    test = PublicExperiments.objects.filter(experiment=experiment)
    if len(test) > 0:
        return True
    else:
        return False


# @login_required(login_url='/registration/login/')
def show_doc(request, doc_id):
    document = Document.objects.get(id=doc_id)
    experiment = document.experiment

    if not check_user(request, experiment):
        return HttpResponse("You don't have permission to access this page")
    print(document.experiment.experiment_type)
    if document.experiment.experiment_type == '0':
        context_dict = get_doc_context_dict(document)
    elif document.experiment.experiment_type == '1':
        context_dict = get_decomp_doc_context_dict(document)
    else:
        context_dict = {}
    print(context_dict)
    context_dict['document'] = document
    context_dict['experiment'] = experiment
    context_dict['is_public'] = is_public(experiment)

    if document.csid:
        context_dict['csid'] = document.csid

    # remove this -- deprecated...
    if document.image_url:
        context_dict['image_url'] = document.image_url

    if not document.mol_string:
        try:
            from chemspipy import ChemSpider
            cs = ChemSpider(settings.CHEMSPIDER_APIKEY)
            md = jsonpickle.decode(document.metadata)
            if 'InChIKey' in md or 'inchikey' in md:
                mol = cs.convert(md.get('InChIKey', md['inchikey']), 'InChIKey', 'mol')
                if mol:
                    document.mol_string = mol
                    document.save()
        except:
            pass

    if document.mol_string:
        context_dict['mol_string'] = document.mol_string
        context_dict['image_url'] = None

    sub_terms = document.substituentinstance_set.all()
    if len(sub_terms) > 0:
        context_dict['sub_terms'] = sub_terms

    return render(request, 'basicviz/show_doc.html', context_dict)


def get_doc_context_dict(document):
    features = FeatureInstance.objects.filter(document=document).select_related(
        'feature'
    ).prefetch_related(
        'featuremass2motifinstance_set',
        'featureinstance2sub_set',
        'feature__experiment',
        'feature__experiment__document_set'
    )
    context_dict = {}
    context_dict['features'] = features
    experiment = document.experiment

    mass2motif_instances = get_docm2m_bydoc(document)
    context_dict['mass2motifs'] = mass2motif_instances
    feature_mass2motif_instances = []
    original_experiment = None
    for feature_instance in features:
        if feature_instance.intensity > 0:
            m2m = feature_instance.featuremass2motifinstance_set.all()
            smiles_to_docs = defaultdict(set)
            docs_to_subs = {}

            # if this experiment already has magma annotation, then we just get the magma annotations
            # of the feature instances in this document. Otherwise we use the global feature to find
            # the magma annotation from the original experiment that has the magma annotation
            if experiment.has_magma_annotation:
                subs = feature_instance.featureinstance2sub_set.all()
                for sub in subs:
                    smiles = sub.sub.smiles
                    smiles_to_docs[smiles].add(document)
                    docs_to_subs[document] = sub
            else:
                # get shared global feature
                shared_feature = feature_instance.feature

                # get original docs having shared feature
                original_experiment = shared_feature.experiment
                original_docs = shared_feature.experiment.document_set.all()

                # get the feature instances in the original docs having shared feature
                original_feature_instances = FeatureInstance.objects.filter(feature=shared_feature,
                                                                            document__in=original_docs)

                # get magma subs from the original feature instances
                subs = FeatureInstance2Sub.objects.filter(feature__in=original_feature_instances).select_related(
                    'sub',
                    'feature',
                    'feature__document'
                )
                for sub in subs:
                    smiles = sub.sub.smiles
                    doc = sub.feature.document
                    smiles_to_docs[smiles].add(sub.feature.document)
                    docs_to_subs[doc] = sub

            # count how many docs are found containing each magma substructure linked to this feature instance
            # if the experiment has_magma_annotation, this should always be one
            smiles_docs_count = Counter()
            for smiles in smiles_to_docs:
                n_docs = len(smiles_to_docs[smiles])
                if experiment.has_magma_annotation:
                    assert n_docs == 1
                smiles_docs_count[smiles] += n_docs

            most_common_subs = []
            for smiles, count in smiles_docs_count.most_common():
                docs = smiles_to_docs[smiles]
                subs = [docs_to_subs[d] for d in docs]
                together = zip(docs, subs)
                most_common_subs.append((smiles, count, together,))
            item = (feature_instance, m2m, most_common_subs, len(most_common_subs),)
            feature_mass2motif_instances.append(item)

    if original_experiment:
        context_dict['original_experiment'] = original_experiment
    context_dict['experiment'] = experiment
    feature_mass2motif_instances = sorted(feature_mass2motif_instances, key=lambda x: x[0].intensity, reverse=True)
    context_dict['fm2m'] = feature_mass2motif_instances
    context_dict['top_n'] = 5  # show the top-5 magma annotations per feature initially
    return context_dict


# @login_required(login_url='/registration/login/')
def view_parents(request, motif_id):
    # use the get_subclass method instead of standard get
    motif = Mass2Motif.objects.get_subclass(id=motif_id)
    print(type(motif))
    if type(motif) == MDBMotif:
        return redirect('/motifdb/motif/{}'.format(motif_id))
    experiment = motif.experiment

    if not check_user(request, experiment):
        return HttpResponse("You don't have permission to access this page")
    print('Motif metadata', motif.metadata)
    context_dict = {'mass2motif': motif}
    motif_feature_instances = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')
    total_prob = sum([m.probability for m in motif_feature_instances])

    # get all the documents linked above threshold to this motif in this experiment
    dm2m = get_docm2m_by_motif(experiment, motif)
    documents = [x.document for x in dm2m]

    # get all substructures linked to features in documents
    motif_features_subs = []
    for motif_feature_instance in motif_feature_instances:
        print('Querying substructures of motif_feature_instance %s' % motif_feature_instance)
        feature = motif_feature_instance.feature
        document_feature_instance = FeatureInstance.objects.filter(feature=feature, document__in=documents)
        docs = set([x.document for x in document_feature_instance])
        subs = FeatureInstance2Sub.objects.filter(feature__in=document_feature_instance)
        most_common = most_common_subs(subs, docs)  # sort subs by most-common
        motif_features_subs.append((motif_feature_instance, most_common, len(most_common),))

    context_dict['motif_features_subs'] = motif_features_subs
    context_dict['top_n'] = 5  # show the top-5 magma annotations per feature initially
    context_dict['total_prob'] = total_prob
    context_dict['experiment'] = experiment

    # Get the taxa or substituent terms (if there are any)
    # taxa_terms = motif.taxainstance_set.filter(probability__gte=0.2).order_by('-probability')
    # substituent_terms = motif.substituentinstance_set.filter(probability__gte=0.2).order_by('-probability')

    # if len(taxa_terms) > 0:
    #     context_dict['taxa_terms'] = taxa_terms
    # if len(substituent_terms) > 0:
    #     context_dict['substituent_terms'] = substituent_terms

    dm2m = get_docm2m(motif)
    context_dict['dm2ms'] = dm2m

    context_dict['status'] = None
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
            context_dict['status'] = 'Annotation saved.'

    permission = check_user(request, experiment)
    if permission == 'edit':
        metadata_form = Mass2MotifMetadataForm(
            initial={'metadata': motif.annotation, 'short_annotation': motif.short_annotation})
        context_dict['metadata_form'] = metadata_form
    else:  # read only permission
        context_dict['motif_annotation'] = motif.annotation
        context_dict['short_annotation'] = motif.short_annotation

    massbank_form = get_massbank_form(motif, motif_feature_instances)
    context_dict['massbank_form'] = massbank_form

    # New classyfire code
    term_counts = {}
    for dm in dm2m:
        doc = dm.document
        sub_terms = doc.substituentinstance_set.all()
        print(doc.experiment)
        for s in sub_terms:
            if not s.subterm in term_counts:
                term_counts[s.subterm] = 0
            term_counts[s.subterm] += 1
    if len(term_counts) > 0:
        # compute overall percentages for this experiment
        totals = {}
        n_docs = len(Document.objects.filter(experiment=experiment))
        temp = {}
        for t in term_counts:
            temp[t] = len(SubstituentInstance.objects.filter(document__experiment=experiment, subterm=t))
            if temp[t] < term_counts[t]:
                print(t)
            totals[t] = 100.0 * len(
                SubstituentInstance.objects.filter(document__experiment=experiment, subterm=t)) / n_docs
        terms = term_counts.keys()
        counts = term_counts.values()
        perc = [(100.0 * v) / len(dm2m) for v in term_counts.values()]
        background = [totals[t] for t in term_counts.keys()]
        diff = [abs(p - b) for (p, b) in zip(perc, background)]
        context_dict['term_counts'] = sorted(zip(terms, counts, perc, background, diff), key=lambda x: x[1],
                                             reverse=True)
    return render(request, 'basicviz/view_parents.html', context_dict)


# sort substuctures by most common
def most_common_subs(subs, docs):
    subs_smiles = []
    subs_dict = defaultdict(set)
    seen_docs = set()
    for s in subs:
        key = s.sub.smiles
        subs_smiles.append(key)
        subs_dict[key].add(s)
        # also track the documents containing this feature instance and annotated with some magma substructure
        feature_doc = s.feature.document
        seen_docs.add(feature_doc)
    subs_counter = Counter(subs_smiles)  # count most common substructures
    most_common = []
    for smile, count in subs_counter.most_common():
        feature_sub_instances = subs_dict[smile]
        most_common.append((smile, count, feature_sub_instances,))
    # add the count for documents which are not annotated with magma substructures
    num_unseen_docs = len(docs - seen_docs)
    most_common.append(('None', num_unseen_docs, set()))
    return most_common


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
    edge_choice = get_option('default_doc_m2m_score', experiment)
    if experiment.experiment_type == '0':  # ms2lda
        motif = Mass2Motif.objects.get(id=motif_id)

        docm2m = get_docm2m(motif)
        # if edge_choice == 'probability':
        #     docm2m = DocumentMass2Motif.objects.filter(mass2motif=motif, probability__gte=viz_options.edge_thresh).order_by(
        #         '-probability')
        # else:
        #     docm2m = DocumentMass2Motif.objects.filter(mass2motif=motif,
        #                                                overlap_score__gte=viz_options.edge_thresh).order_by(
        #         '-overlap_score')
        documents = [d.document for d in docm2m]
        parent_data = []
        for dm in docm2m:
            document = dm.document
            parent_data.append(get_doc_for_plot(document.id, motif_id, score_type=edge_choice))
    else:  # decomposition
        parent_data = get_parents_decomposition(motif_id, vo_id=vo_id, experiment=experiment)
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
def get_all_parents_metadata(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    documents = Document.objects.filter(experiment=experiment)
    parent_data = []
    for document in documents:
        parent_data.append((document.name, jsonpickle.decode(document.metadata)))
    return HttpResponse(json.dumps(parent_data), content_type='application/json')


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

def get_word_graph(request, motif_id, vo_id, experiment=None):
    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id=vo_id)
        experiment = viz_options.experiment
    else:
        motif = Mass2Motif.objects.get(id=motif_id)
        experiment = motif.experiment

    if experiment.experiment_type == "0":  # standard LDA
        motif = Mass2Motif.objects.get(id=motif_id)
        m2mIns = Mass2MotifInstance.objects.filter(mass2motif=motif, probability__gte=0.01)
        docm2ms = get_docm2m(motif)
        # if edge_choice == 'probability':
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh)
        # else:
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh)
        data_for_json = []
        data_for_json.append(len(docm2ms))
        feat_counts = {}
        for feature in m2mIns:
            feat_counts[feature.feature] = 0
        for dm2m in docm2ms:
            fi = FeatureInstance.objects.filter(document=dm2m.document)
            for ft in fi:
                if ft.feature in feat_counts:
                    feat_counts[ft.feature] += 1
        colours = '#404080'
        feat_list = []
        for feature in feat_counts:
            feat_type = feature.name.split('_')[0]
            try:
                feat_mz = feature.name.split('_')[1]
                short_name = "{}_{:.4f}".format(feat_type, float(feat_mz))
            except:
                short_name = feature.name
            feat_list.append([short_name, feat_counts[feature], colours])
        feat_list = sorted(feat_list, key=lambda x: x[1], reverse=True)
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


def get_intensity(request, motif_id, vo_id, experiment=None):
    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id=vo_id)
        experiment = viz_options.experiment
    else:
        motif = Mass2Motif.objects.get(id=motif_id)
        experiment = motif.experiment

    colours = ['#404080', '#0080C0']
    colours = ['red', 'blue']

    if experiment.experiment_type == "0":  # standard LDA
        motif = Mass2Motif.objects.get(id=motif_id)
        m2mIns = Mass2MotifInstance.objects.filter(mass2motif=motif, probability__gte=0.01)
        docm2ms = get_docm2m(motif)
        # if edge_choice == 'probability':
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = edge_thresh)
        # else:
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif = motif,overlap_score__gte = edge_thresh)
        documents = [d.document for d in docm2ms]
        data_for_json = []
        feat_total_intensity = {}
        feat_motif_intensity = {}
        features = [m.feature for m in m2mIns]
        for feature in features:
            feat_total_intensity[feature] = 0.0
            feat_motif_intensity[feature] = 0.0
        for feature in features:
            fi = FeatureInstance.objects.filter(feature=feature)
            for ft in fi:
                feat_total_intensity[feature] += ft.intensity
                if ft.document in documents:
                    feat_motif_intensity[feature] += ft.intensity

        feat_list = []
        feat_tot_intensity = zip(feat_total_intensity.keys(), feat_total_intensity.values())
        feat_tot_intensity = sorted(feat_tot_intensity, key=lambda x: x[1], reverse=True)
        for feature, tot_intensity in feat_tot_intensity:
            feat_type = feature.name.split('_')[0]
            try:
                feat_mz = feature.name.split('_')[1]
                short_name = "{}_{:.4f}".format(feat_type, float(feat_mz))
            except:
                short_name = feature.name
            feat_list.append([short_name, feat_total_intensity[feature], colours[0]])
            feat_list.append(['', feat_motif_intensity[feature], colours[1]])
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
    if not check_user(request, experiment):
        return HttpResponse("You do not have permission to access this page")
    if experiment.experiment_type == '0':
        motif_tuples = get_motifs_with_degree(experiment)
        context_dict = {'motif_tuples': motif_tuples}
        context_dict['experiment'] = experiment
        return render(request, 'basicviz/view_mass2motifs.html', context_dict)
    elif experiment.experiment_type == '1':  # decomp
        raise Http404('Page not found')
        # documents = Document.objects.filter(experiment = experiment)
        # dm2m = DocumentGlobalMass2Motif.objects.filter(document__in = documents)
        # mass2motifs = list(set([d.mass2motif for d in dm2m]))
        # context_dict = {'mass2motifs':mass2motifs,'experiment':experiment}
        return render(request, 'decomposition/view_mass2motifs.html', context_dict)


def get_doc_for_plot(doc_id, motif_id=None, get_key=False, score_type=None):
    colours = ['red', 'green', 'black', 'orange']
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document=document)
    plot_fragments = []

    # Get the parent info
    metadata = jsonpickle.decode(document.metadata)
    precursor_mass = document.mass
    # if 'parentmass' in metadata:
    #     parent_mass = float(metadata['parentmass'])
    # elif 'mz' in metadata:
    #     parent_mass = float(metadata['mz'])
    # elif '_' in document.name:
    #     try:
    #         parent_mass = float(document.name.split('_')[0])
    #     except:
    #         # in case the name isnt that format
    #         parent_mass = 0.0
    # else:
    #     parent_mass = 0.0
    probability = "na"

    # default_score = get_option('default_doc_m2m_score', experiment=document.experiment)
    # if not default_score:
    #     default_score = 'probability'

    # following is only used now when we're getting the multi-colour plot
    default_score = 'probability'

    if not motif_id == None:
        m2m = Mass2Motif.objects.get(id=motif_id)
        dm2m = DocumentMass2Motif.objects.get(mass2motif=m2m, document=document)
        probability = "Probability: {}, overlap: {}".format(dm2m.probability, dm2m.overlap_score)
        # if not score_type:
        #     if default_score == 'probability':
        #         probability = dm2m.probability
        #     else:
        #         probability = dm2m.overlap_score
        # else:
        #     if score_type == 'probability':
        #         probability = dm2m.probability
        #     else:
        #         probability = dm2m.overlap_score

    parent_data = (precursor_mass, 100.0, document.display_name, document.annotation, probability)
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

        # find positions of the diffs
        diff_feature_names = [f.feature.name for f in features if f.feature.name.startswith('mzdiff')]
        frag_feature_names = [f.feature.name for f in features if f.feature.name.startswith('fragment')]
        frag_intensities = [f.intensity for f in features if f.feature.name.startswith('fragment')]
        diff_masses = [f.split('_')[1] for f in diff_feature_names]
        frag_masses = [f.split('_')[1] for f in frag_feature_names]
        diff_instances = {}

        for d in diff_masses:
            diff_instances[d] = []
            temp = ["{:.4f}".format(float(d) + float(f)) for f in frag_masses]
            # temp is a vector of length number frag features, including all the frags transformed by this diff

            diff_instances[d] = [(i, frag_masses.index(t)) for i, t in enumerate(temp) if t in frag_masses]
        print(diff_instances)

        diff_pos = 100

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
            elif feature_name.startswith('loss'):
                cum_pos = precursor_mass - mass
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
                    (precursor_mass - other_topics, precursor_mass, this_intensity, this_intensity, 0, 'gray',
                     feature_name))
            elif feature_name.startswith('mzdiff'):
                # note we only plot them in the correct colour if they have phi > 0.5
                diff_mass = feature_name.split('_')[1]
                if len(diff_instances[diff_mass]) > 0:
                    for start, stop in diff_instances[diff_mass]:
                        start_mass = frag_masses[start]
                        stop_mass = frag_masses[stop]
                        print(start_mass, stop_mass)
                        intensity = min(frag_intensities[start], frag_intensities[stop]) * 100.0 / max_intensity
                        plotted = False
                        for phi_value in phi_values:
                            if phi_value.mass2motif in topics_to_plot and phi_value.probability >= 0.5:
                                colour = topic_colours[phi_value.mass2motif]
                                child_data.append((
                                                  float(start_mass), float(stop_mass), 0.9 * intensity, 0.9 * intensity,
                                                  2, colour, feature_name)
                                                  )
                                diff_pos -= 5
                                plotted = True
                        if not plotted:
                            # doesn't belong to any of the dominant topics
                            child_data.append((float(start_mass), float(stop_mass), 0.9 * intensity, 0.9 * intensity, 2,
                                               'gray', feature_name)
                                              )

    plot_fragments.append(child_data)

    if get_key:
        key = []
        for topic in topic_colours:
            key.append((topic.name, topic_colours[topic]))
        return [plot_fragments], key

    return plot_fragments


def get_doc_topics(request, doc_id):
    document = Document.objects.get(id=doc_id)
    if document.experiment.experiment_type == '0':
        plot_fragments = [get_doc_for_plot(doc_id, get_key=True)]
    elif document.experiment.experiment_type == '1':  # decomposition
        raise Http404('Page not found')
        # score_type = get_option('default_doc_m2m_score',experiment = document.experiment)
        # if not score_type:
        #     score_type = 'probability'
        # plot_fragments = [get_parent_for_plot_decomp(document,edge_choice=score_type,get_key = True)]
    else:
        plot_fragments = []
    return HttpResponse(json.dumps(plot_fragments), content_type='application/json')


# @login_required(login_url='/registration/login/')
def start_viz(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {'experiment': experiment}

    ## Only show analysis choices done lthrough celery
    ready, _ = EXPERIMENT_STATUS_CODE[1]
    choices = [(analysis.id, analysis.name + '(' + analysis.description + ')') for analysis in
               Analysis.objects.filter(experiment=experiment, status=ready)]
    if request.method == 'POST':
        viz_form = VizForm(choices, request.POST)
        if viz_form.is_valid():
            min_degree = viz_form.cleaned_data['min_degree']
            # edge_thresh = viz_form.cleaned_data['edge_thresh']
            # j_a_n = viz_form.cleaned_data['just_annotated_docs']
            # colour_by_logfc = viz_form.cleaned_data['colour_by_logfc']
            # discrete_colour = viz_form.cleaned_data['discrete_colour']
            # lower_colour_perc = viz_form.cleaned_data['lower_colour_perc']
            # upper_colour_perc = viz_form.cleaned_data['upper_colour_perc']
            # colour_topic_by_score = viz_form.cleaned_data['colour_topic_by_score']
            # random_seed = viz_form.cleaned_data['random_seed']
            # edge_choice = viz_form.cleaned_data['edge_choice']
            # edge_choice = edge_choice[0].encode('ascii', 'ignore')  # should turn the unicode into ascii
            ## do not colour document nodes analysis has not been chosen or use the default empty ('') analysis
            if len(viz_form.cleaned_data['ms1_analysis']) == 0 or viz_form.cleaned_data['ms1_analysis'][0] == '':
                ms1_analysis_id = None
            else:
                ms1_analysis_id = viz_form.cleaned_data['ms1_analysis'][0]
            vo = VizOptions.objects.get_or_create(experiment=experiment,
                                                  min_degree=min_degree,
                                                  # edge_thresh=edge_thresh,
                                                  # just_annotated_docs=j_a_n,
                                                  # colour_by_logfc=colour_by_logfc,
                                                  # discrete_colour=discrete_colour,
                                                  # lower_colour_perc=lower_colour_perc,
                                                  # upper_colour_perc=upper_colour_perc,
                                                  # colour_topic_by_score=colour_topic_by_score,
                                                  # random_seed=random_seed,
                                                  # edge_choice=edge_choice,
                                                  ms1_analysis_id=ms1_analysis_id)[0]
            context_dict['viz_options'] = vo

        else:
            context_dict['viz_form'] = viz_form
    else:
        viz_form = VizForm(choices)
        context_dict['viz_form'] = viz_form

    if 'viz_form' in context_dict:
        return render(request, 'basicviz/viz_form.html', context_dict)
    else:
        # initial_motif = Mass2Motif.objects.filter(experiment=experiment)[0]
        # context_dict['initial_motif'] = initial_motif
        return render(request, 'basicviz/graph.html', context_dict)


# @login_required(login_url='/registration/login/')
def start_annotated_viz(request, experiment_id):
    # Is this function ever called??
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
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
    show_ms1 = request.GET.get('show_ms1')
    if show_ms1 == 'true':
        show_ms1 = True
    else:
        show_ms1 = False

    if experiment.experiment_type == "0":
        ms1_analysis_id = viz_options.ms1_analysis_id if show_ms1 else None
        G = make_graph(experiment, min_degree=viz_options.min_degree,
                       # edge_thresh=viz_options.edge_thresh,
                       # just_annotated_docs=viz_options.just_annotated_docs,
                       # colour_by_logfc=viz_options.colour_by_logfc,
                       # discrete_colour=viz_options.discrete_colour,
                       # lower_colour_perc=viz_options.lower_colour_perc,
                       # upper_colour_perc=viz_options.upper_colour_perc,
                       # colour_topic_by_score=viz_options.colour_topic_by_score,
                       # edge_choice=viz_options.edge_choice,
                       ms1_analysis_id=ms1_analysis_id)
    else:
        # G = make_decomposition_graph(experiment, min_degree=viz_options.min_degree,
        #                edge_thresh=viz_options.edge_thresh,
        #                edge_choice=viz_options.edge_choice)
        raise Http404("page not found")
    d = json_graph.node_link_data(G)

    # convert links from nodes ids to node indices
    print('Converting links to indices')
    node_indices = {d['nodes'][i]['id']: i for i in range(len(d['nodes']))}
    new_links = []
    for link in d['links']:
        source_id = node_indices[link['source']]
        target_id = node_indices[link['target']]
        weight = link['weight']
        new_link = {
            'source': source_id,
            'target': target_id,
            'weight': weight
        }
        new_links.append(new_link)
    d['links'] = new_links
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


# def make_graph(experiment, edge_thresh=0.05, min_degree=5,
#                topic_scale_factor=5, edge_scale_factor=5, just_annotated_docs=False,
#                colour_by_logfc=False, discrete_colour=False, lower_colour_perc=10, upper_colour_perc=90,
#                colour_topic_by_score=False, edge_choice='probability', ms1_analysis_id = None, doc_max_size = 200, motif_max_size = 1000):
def make_graph(experiment, min_degree=5, topic_scale_factor=5, edge_scale_factor=5,
               ms1_analysis_id=None, doc_max_size=200, motif_max_size=1000):
    mass2motifs = Mass2Motif.objects.filter(experiment=experiment)
    documents = Document.objects.filter(experiment=experiment)
    doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)
    docm2ms_q = DocumentMass2Motif.objects.filter(mass2motif__experiment=experiment,
                                                  probability__gte=doc_m2m_prob_threshold,
                                                  overlap_score__gte=doc_m2m_overlap_threshold) \
        .select_related('document').select_related('mass2motif')
    docm2ms = {}
    for r in docm2ms_q:
        if r.mass2motif_id in docm2ms:
            docm2ms[r.mass2motif_id].append(r)
        else:
            docm2ms[r.mass2motif_id] = [r]
    # Find the degrees
    topics = {}
    docm2m_dict = {}
    for mass2motif in mass2motifs:
        if mass2motif.id in docm2ms:
            docm2m_dict[mass2motif] = docm2ms[mass2motif.id]
        else:
            docm2m_dict[mass2motif] = []
        topics[mass2motif] = len(docm2m_dict[mass2motif])

        # if edge_choice == 'probability':
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=edge_thresh)
        # else:
        #     docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, overlap_score__gte=edge_thresh)
    to_remove = []
    for topic in topics:
        if topics[topic] < min_degree:
            to_remove.append(topic)
    for topic in to_remove:
        del topics[topic]

    docm2mset = []
    for topic in topics:
        docm2mset += docm2m_dict[topic]

    # if edge_choice == 'probability':
    #     docm2mset = DocumentMass2Motif.objects.filter(document__in=documents, mass2motif__in=topics,
    #                                                   probability__gte=edge_thresh)
    # else:
    #     docm2mset = DocumentMass2Motif.objects.filter(document__in=documents, mass2motif__in=topics,
    #                                                   overlap_score__gte=edge_thresh)

    ## remove dependence on "colour nodes by logfc" and "discrete colouring"
    ## document colouring and size setting only depends on users' choice of ms1 analysis setting
    do_plage_flag = True
    if ms1_analysis_id:
        analysis = Analysis.objects.filter(id=ms1_analysis_id)[0]
        all_logfc_vals = []
        res = AnalysisResult.objects.filter(analysis=analysis, document__in=[docm2m.document for docm2m in docm2mset])
        for analysis_result in res:
            foldChange = analysis_result.foldChange
            logfc = np.log(foldChange)
            if not np.abs(logfc) == np.inf:
                all_logfc_vals.append(np.log(foldChange))
        min_logfc = np.min(all_logfc_vals)
        max_logfc = np.max(all_logfc_vals)

        ## try make graph for plage
        all_plage_vals = []
        for plage_result in AnalysisResultPlage.objects.filter(analysis=analysis, mass2motif__in=topics.keys()):
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
        metadata = jsonpickle.decode(topic.metadata)
        # if colour_topic_by_score:
        #     upscore = metadata.get('upscore', 1.0)
        #     downscore = metadata.get('downscore', 1.0)
        #     if upscore < 0.05:
        #         highlight_colour = '#0000FF'
        #     elif downscore < 0.05:
        #         highlight_colour = '#FF0000'
        #     else:
        #         highlight_colour = '#AAAAAA'
        #     name = metadata.get('annotation', topic.name)
        #     G.add_node(topic.name, group=2, name=name,
        #                size=topic_scale_factor * topics[topic],
        #                special=True, in_degree=topics[topic],
        #                score=1, node_id=topic.id, is_topic=True,
        #                highlight_colour=highlight_colour)

        ## try make graph for plage
        if ms1_analysis_id and do_plage_flag:
            ## white to green
            lowcol = [255, 255, 255]
            endcol = [0, 255, 0]
            plage_result = AnalysisResultPlage.objects.filter(analysis=analysis, mass2motif=topic)[0]
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
            na = topic.short_annotation
            if na:
                na += ' (' + topic.name + ')'
            else:
                na = topic.name
            G.add_node(topic.name, group=2, name=na + ", " + str(plage_t_value) + ", " + str(plage_p_value),
                       # size=topic_scale_factor * topics[topic],
                       size=size,
                       special=True, in_degree=topics[topic],
                       highlight_colour=col,
                       score=1, node_id=topic.id, is_topic=True)

        else:
            if topic.short_annotation:
                # if 'annotation' in metadata:
                G.add_node(topic.name, group=2, name=topic.short_annotation,
                           size=topic_scale_factor * topics[topic],
                           special=True, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)
            else:
                G.add_node(topic.name, group=2, name=topic.name,
                           size=topic_scale_factor * topics[topic],
                           special=False, in_degree=topics[topic],
                           score=1, node_id=topic.id, is_topic=True)

    # if just_annotated_docs:
    #     new_documents = []
    #     for document in documents:
    #         if document.annotation:
    #             new_documents.append(document)

    #     documents = new_documents

    doc_nodes = []

    print("Second")

    # edge_choice = get_option('default_doc_m2m_score',experiment)
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
                analysis_result = AnalysisResult.objects.filter(analysis=analysis, document=docm2m.document)[0]
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
                        pos = logfc / min_logfc
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


@login_required(login_url='/registration/login/')
def topic_pca(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {'experiment': experiment}
    url = '/basicviz/get_topic_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request, 'basicviz/pca.html', context_dict)


@login_required(login_url='/registration/login/')
def document_pca(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {}
    context_dict['experiment'] = experiment
    url = '/basicviz/get_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request, 'basicviz/pca.html', context_dict)


def get_topic_pca_data(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)

    # features = Feature.objects.filter(experiment=experiment)
    documents = Document.objects.filter(experiment=experiment)
    featureinstance = FeatureInstance.objects.filter(document__in=documents)
    features = set([f.feature for f in featureinstance])

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
    if not check_user(request, experiment):
        return HttpResponse("You do not have permission to access this page")
    context_dict = {}
    if request.method == 'POST':
        form = ValidationForm(request.POST)
        if form.is_valid():
            p_thresh = form.cleaned_data['p_thresh']
            overlap_thresh = form.cleaned_data['overlap_thresh']
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
                    dm2ms = get_docm2m(mass2motif, doc_m2m_prob_threshold=p_thresh,
                                       doc_m2m_overlap_threshold=overlap_thresh)
                    tot = 0
                    val = 0
                    for d in dm2ms:
                        if (just_annotated and d.document.annotation) or not just_annotated:
                            tot += 1
                            if d.validated:
                                val += 1
                    counts.append((tot, val))
                    all_dm2ms.append(dm2ms)
                    print(dm2ms)
            annotated_mass2motifs = zip(annotated_mass2motifs, counts, all_dm2ms)
            context_dict['annotated_mass2motifs'] = annotated_mass2motifs
            context_dict['counts'] = counts
            context_dict['p_thresh'] = p_thresh
            context_dict['overlap_thresh'] = overlap_thresh
            context_dict['just_annotated'] = just_annotated

        else:
            context_dict['validation_form'] = form
    else:

        form = ValidationForm()
        context_dict['validation_form'] = form
    context_dict['experiment'] = experiment
    return render(request, 'basicviz/validation.html', context_dict)


def toggle_dm2m(request, experiment_id, dm2m_id):
    permission = check_user(request, experiment)
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
        ['m2m_id', 'm2m_name', 'm2m_annotation', 'doc_id', 'doc_annotation', 'valid', 'probability', 'overlap_score',
         'doc_csid',
         'doc_inchi'])

    dm2ms = get_docm2m(mass2motif)
    for dm2m in dm2ms:
        document = dm2m.document
        # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
        doc_name = '"' + dm2m.document.display_name + '"'
        annotation = '"' + mass2motif.annotation + '"'
        writer.writerow([mass2motif.id, mass2motif.name, mass2motif.annotation.encode('utf8'), dm2m.document.id,
                         doc_name.encode('utf8'), dm2m.validated, dm2m.probability, dm2m.overlap_score,
                         dm2m.document.csid,
                         dm2m.document.inchikey])

    return response


## Refactored on Oct 11th, 2017
## to be re-used by different docm2m fetching methods
def get_prob_overlap_thresholds(experiment, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    ## default prob_threshold 0.05, default overlap_threshld 0.0
    if not doc_m2m_prob_threshold:
        doc_m2m_prob_threshold = get_option('doc_m2m_prob_threshold', experiment=experiment)
        if doc_m2m_prob_threshold:
            doc_m2m_prob_threshold = float(doc_m2m_prob_threshold)
        else:
            doc_m2m_prob_threshold = 0.05

    if not doc_m2m_overlap_threshold:
        doc_m2m_overlap_threshold = get_option('doc_m2m_overlap_threshold', experiment=experiment)
        if doc_m2m_overlap_threshold:
            doc_m2m_overlap_threshold = float(doc_m2m_overlap_threshold)
        else:
            doc_m2m_overlap_threshold = 0.0

    return doc_m2m_prob_threshold, doc_m2m_overlap_threshold


## updated get_docm2m function, use threshold for probability and overlap respectively
## function is used to get MocumentMassMass2Motif by motif only
def get_docm2m(mass2motif, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    if doc_m2m_prob_threshold is None or doc_m2m_overlap_threshold is None:
        experiment = mass2motif.experiment
        doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)

    dm2m = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=doc_m2m_prob_threshold,
                                             overlap_score__gte=doc_m2m_overlap_threshold).order_by('-probability')

    return dm2m


## function is used to get all MocumentMassMass2Motif matchings by experiment
def get_docm2m_all(experiment, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)

    mass2motifs = Mass2Motif.objects.filter(experiment=experiment)

    dm2m = DocumentMass2Motif.objects.filter(mass2motif__in=mass2motifs, probability__gte=doc_m2m_prob_threshold,
                                             overlap_score__gte=doc_m2m_overlap_threshold).order_by(
        '-probability').select_related('mass2motif').prefetch_related('document')

    return dm2m


## function is used to get all MocumentMassMass2Motif matchings by experiment and mass2motif
def get_docm2m_by_motif(experiment, mass2motif, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)
    dm2m = DocumentMass2Motif.objects.filter(mass2motif=mass2motif, probability__gte=doc_m2m_prob_threshold,
                                             overlap_score__gte=doc_m2m_overlap_threshold).order_by('-probability')

    return dm2m


## function is used to get MocumentMassMass2Motif by document only
def get_docm2m_bydoc(document, doc_m2m_prob_threshold=None, doc_m2m_overlap_threshold=None):
    experiment = document.experiment

    doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)

    dm2m = DocumentMass2Motif.objects.filter(document=document, probability__gte=doc_m2m_prob_threshold,
                                             overlap_score__gte=doc_m2m_overlap_threshold).order_by('-probability')

    return dm2m


# @login_required(login_url='/registration/login/')
def extract_docs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    if not check_user(request, experiment):
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
        featureinstance__in=document_feature_instances).select_related('featureinstance__feature')
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
    if not check_user(request, experiment):
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
def high_classyfire(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)
    taxa_instances = TaxaInstance.objects.filter(motif__in=motifs, probability__gte=0.2)
    substituent_instances = SubstituentInstance.objects.filter(motif__in=motifs, probability__gte=0.2)
    context_dict = {}
    context_dict['taxa_instances'] = taxa_instances
    context_dict['substituent_instances'] = substituent_instances
    context_dict['experiment'] = experiment
    return render(request, 'basicviz/high_classyfire.html', context_dict)


def get_features(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    documents = Document.objects.filter(experiment=experiment)
    feature_instances = FeatureInstance.objects.filter(document__in=documents)
    features = set([f.feature for f in feature_instances])
    # features = Feature.objects.filter(experiment = experiment)
    output_features = [(f.name, f.min_mz, f.max_mz) for f in features]
    return HttpResponse(json.dumps(output_features), content_type='application/json')


def get_document(request):
    return_val = {}
    # try:
    # todo: add check for public experiment
    experiment_id = request.GET['experiment_id']
    document_id = request.GET['document_id']
    try:
        experiment = Experiment.objects.get(id=int(experiment_id))
        if is_public(experiment):

            return_val['experiment_name'] = experiment.name
            try:
                document = Document.objects.get(id=int(document_id))
                fi = FeatureInstance.objects.filter(document=document)
                peaks = []
                for f in fi:
                    if f.feature.name.startswith('fragment'):
                        mz = float(f.feature.name.split('_')[1])
                        intensity = f.intensity
                        peaks.append((mz, intensity))
                return_val['peaks'] = peaks
                return_val['precursor_mz'] = document.get_mass()
            except:
                return_val['error'] = "Document {} does not exist".format(document_id)
        else:
            return_val['error'] = "This feature only works for public experiments"
    except:
        return_val['error'] = "Experiment {} does not exist".format(experiment_id)

    # except:
    #     return_val['status'] = 'failed'
    return HttpResponse(json.dumps(return_val), content_type='application/json')


def get_annotated_topics(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)
    output_motifs = []
    for motif in motifs:
        if motif.annotation:
            output_motifs.append(motif)

    output_metadata = []
    output_beta = []
    for motif in output_motifs:
        output_metadata.append((motif.name, motif.annotation, motif.short_annotation))
        betas = []
        beta_vals = motif.mass2motifinstance_set.all()
        for b in beta_vals:
            betas.append((b.feature.name, b.probability))
        output_beta.append((motif.name, betas))

    output = (output_metadata, output_beta)

    return HttpResponse(json.dumps(output), content_type='application/json')


def get_all_topics(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    motifs = Mass2Motif.objects.filter(experiment=experiment)
    output_motifs = motifs

    output_metadata = []
    output_beta = []
    for motif in output_motifs:
        output_metadata.append((motif.name, motif.annotation, motif.short_annotation))
        betas = []
        beta_vals = motif.mass2motifinstance_set.all()
        for b in beta_vals:
            betas.append((b.feature.name, b.probability))
        output_beta.append((motif.name, betas))

    output = (output_metadata, output_beta)

    return HttpResponse(json.dumps(output), content_type='application/json')


# Gets the document <-> m2m links for a particular experiment as a json object
def get_doc_m2m(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    dm2m = DocumentMass2Motif.objects.filter(document__experiment=experiment)
    output_data = []
    for d in dm2m:
        output_data.append([d.mass2motif.name, d.document.name, d.probability, d.overlap_score])
    return HttpResponse(json.dumps(output_data), content_type='application/json')


def get_beta(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    mi = Mass2MotifInstance.objects.filter(mass2motif__experiment=experiment)
    output_data = []
    for m in mi:
        output_data.append([m.mass2motif.name, m.feature.name, m.probability])
    return HttpResponse(json.dumps(output_data), content_type='application/json')


def get_all_doc_data(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    documents = Document.objects.filter(experiment=experiment)
    out_data = []
    for document in documents:
        doc_feat = FeatureInstance.objects.filter(document=document)
        doc_features = [(d.feature.name, d.intensity) for d in doc_feat]
        doc_motif = DocumentMass2Motif.objects.filter(document=document)
        doc_motifs = [(d.mass2motif.name, d.probability, d.overlap_score) for d in doc_motif]
        out_data.append([document.name, doc_features, doc_motifs])
    return HttpResponse(json.dumps(out_data), content_type='application/json')


def get_proportion_annotated_docs(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    output_data = []
    documents = Document.objects.filter(experiment=experiment)
    n_docs = len(documents)
    n_annotated = 0
    for document in documents:
        dm2ms = get_docm2m_bydoc(document)
        for dm in dm2ms:
            if dm.mass2motif.annotation:
                n_annotated += 1
                break
    output_data.append((experiment.name, n_docs, n_annotated))
    return HttpResponse(json.dumps(output_data), content_type='application/json')


def get_motifs_with_degree(experiment):
    doc_m2m_prob_threshold, doc_m2m_overlap_threshold = get_prob_overlap_thresholds(experiment)
    motifs = Mass2Motif.objects.filter(experiment=experiment).prefetch_related('experiment')
    docm2m_q = DocumentMass2Motif.objects.values_list('mass2motif__id').filter(mass2motif__experiment=experiment,
                                                                               probability__gte=doc_m2m_prob_threshold,
                                                                               overlap_score__gte=doc_m2m_overlap_threshold,
                                                                               ).annotate(degree=Count('*'))
    docm2m = {r[0]: r[1] for r in docm2m_q}

    motif_tuples = []
    for motif in motifs:
        if motif.id in docm2m:
            motif_tuples.append((motif, docm2m[motif.id]))
        else:
            motif_tuples.append((motif, 0))
    return motif_tuples


# Renders a page summarising a particular experiment
def summary(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    user_experiments = UserExperiment.objects.filter(experiment=experiment)
    this_permission = check_user(request, experiment)
    if not this_permission:
        return HttpResponse("You don't have permission to access this page")
    motif_tuples = get_motifs_with_degree(experiment)

    motif_features = Mass2MotifInstance.objects.filter(mass2motif__experiment=experiment,
                                                       probability__gte=0.05).select_related(
        'mass2motif').prefetch_related('feature')

    documents = Document.objects.filter(experiment=experiment)

    all_docs_motifs = get_docm2m_all(experiment=experiment)

    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['user_experiments'] = user_experiments
    context_dict['motif_tuples'] = motif_tuples
    context_dict['motif_features'] = motif_features
    context_dict['documents'] = documents
    context_dict['n_docs'] = len(documents)
    context_dict['all_docs_motifs'] = all_docs_motifs
    if this_permission == 'edit':
        context_dict['edit_user'] = True
    else:
        context_dict['edit_user'] = False

    pe = PublicExperiments.objects.filter(experiment=experiment)
    if len(pe) > 0:
        context_dict['is_public'] = True
    else:
        context_dict['is_public'] = False

    return render(request, 'basicviz/summary.html', context_dict)


def short_summary(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    user_experiments = UserExperiment.objects.filter(experiment=experiment)

    motifs = Mass2Motif.objects.filter(experiment=experiment)
    motif_tuples = []
    for motif in motifs:
        dm2ms = get_docm2m(motif)
        motif_tuples.append((motif, len(dm2ms)))

    # motif_features = Mass2MotifInstance.objects.filter(mass2motif__experiment=experiment, probability__gte=0.05)

    documents = Document.objects.filter(experiment=experiment)

    # all_docs_motifs = get_docm2m_all(experiment=experiment)

    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['user_experiments'] = user_experiments
    context_dict['motif_tuples'] = motif_tuples
    # context_dict['motif_features'] = motif_features
    context_dict['documents'] = documents
    context_dict['n_docs'] = len(documents)
    # context_dict['all_docs_motifs'] = all_docs_motifs

    return render(request, 'basicviz/short_summary.html', context_dict)


# Matches motifs in one experiment with those in another
# TODO: move to celery, create form
def start_match_motifs(request, experiment_id):
    context_dict = {}
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict['experiment'] = experiment
    if request.method == 'POST':
        match_motif_form = MatchMotifForm(request.user, request.POST)
        if match_motif_form.is_valid():
            base_experiment = match_motif_form.cleaned_data['other_experiment']
            base_experiment_id = base_experiment.id
            minimum_score_to_save = float(match_motif_form.cleaned_data['min_score_to_save'])
            # match_motifs.delay(experiment.id, base_experiment_id, min_score_to_save=minimum_score_to_save)
            match_motifs_set.delay(experiment.id, base_experiment.id, min_score_to_save=minimum_score_to_save)
            return manage_motif_matches(request, experiment_id)
    else:
        match_motif_form = MatchMotifForm(request.user)
    context_dict['match_motif_form'] = match_motif_form
    return render(request, 'basicviz/start_match_motifs.html', context_dict)


def manage_motif_matches(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    matches = MotifMatch.objects.filter(frommotif__experiment=experiment)
    context_dict = {}
    context_dict['matches'] = matches
    context_dict['experiment'] = experiment
    return render(request, 'basicviz/match_motifs.html', context_dict)


def add_link(request, from_motif_id, to_motif_id):
    if not from_motif_id == to_motif_id:
        from_motif = Mass2Motif.objects.get(id=from_motif_id)
        to_motif = Mass2Motif.objects.get(id=to_motif_id)
        from_motif.linkmotif = to_motif
        from_motif.save()
    experiment_id = from_motif.experiment.id
    return manage_motif_matches(request, experiment_id)


def remove_link(request, from_motif_id):
    from_motif = Mass2Motif.objects.get(id=from_motif_id)
    from_motif.linkmotif = None
    from_motif.save()
    experiment_id = from_motif.experiment.id
    return manage_motif_matches(request, experiment_id)


def feature_info(request, feature_id, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    feature = Feature.objects.get(id=feature_id)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['feature'] = feature
    documents = Document.objects.filter(experiment=experiment)
    instances = FeatureInstance.objects.filter(feature=feature, document__in=documents)
    context_dict['n_instances'] = len(instances)
    context_dict['instances'] = instances

    motifs = Mass2Motif.objects.filter(experiment=experiment)
    motif_instances = Mass2MotifInstance.objects.filter(feature=feature, mass2motif__in=motifs)

    context_dict['n_motif_instances'] = len(motif_instances)
    context_dict['motif_instances'] = motif_instances

    print(len(instances))
    return render(request, 'basicviz/feature_info.html', context_dict)


@csrf_exempt
def get_doc_annotation(request):
    response = {}
    if request.method == 'POST':
        experiment_name = request.POST['experiment_name']
        try:
            experiment = Experiment.objects.get(name=experiment_name)
        except:
            response['status'] = 'invalid experiment name'
            return HttpResponse(json.dumps(response), content_type='application/json')
        document_name = request.POST['document_name']
        try:
            document = Document.objects.get(name=document_name, experiment=experiment)
        except:
            response['status'] = 'invalid document name'
            return HttpResponse(json.dumps(response), content_type='application/json')
        annotation = jsonpickle.decode(document.metadata).get('annotation', None)
        response['status'] = 'ok'
        response['annotation'] = annotation
    else:
        response['status'] = 'not a post request'
    return HttpResponse(json.dumps(response), content_type='application/json')


@csrf_exempt
def set_doc_annotation(request):
    response = {}
    if request.method == 'POST':
        experiment_name = request.POST['experiment_name']
        try:
            experiment = Experiment.objects.get(name=experiment_name)
        except:
            response['status'] = 'invalid experiment name'
            return HttpResponse(json.dumps(response), content_type='application/json')
        document_name = request.POST['document_name']
        try:
            document = Document.objects.get(name=document_name, experiment=experiment)
        except:
            response['status'] = 'invalid document name'
            return HttpResponse(json.dumps(response), content_type='application/json')
        # document = Document.objects.get(experiment = experiment,name = document_name)

        annotation = request.POST.get('annotation', None)

        metadata = jsonpickle.decode(document.metadata)
        metadata['annotation'] = annotation
        document.metadata = jsonpickle.encode(metadata)
        document.save()
        response['status'] = 'ok'
    else:
        response['status'] = 'not a post request'
    return HttpResponse(json.dumps(response), content_type='application/json')


def get_gnps_summary(request, experiment_id, metadata_columns=['scans', 'precursormass', 'parentrt']):
    experiment = Experiment.objects.get(id=experiment_id)
    documents = Document.objects.filter(experiment=experiment)
    dm2m = DocumentMass2Motif.objects.filter(document__in=documents).order_by('document')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gnps_output_{}.csv"'.format(experiment_id)
    writer = csv.writer(response)

    writer.writerow(metadata_columns + ['document', 'motif', 'probability', 'overlap'])
    for d in dm2m:
        md = []
        for m in metadata_columns:
            temp = jsonpickle.decode(d.document.metadata)
            val = temp.get(m, 'NA')
            if val == 'NA' and m == 'precursormass':
                val = temp.get('parentmass', 'NA')
            md.append(val)
        writer.writerow(md + [d.document, d.mass2motif, d.probability, d.overlap_score])
    return response


def toggle_public(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    permission = check_user(request, experiment)
    if not permission == 'edit':
        return HttpResponse("You don't have the permission to do this!")
    else:
        pe = PublicExperiments.objects.filter(experiment=experiment)
        if len(pe) == 0:
            # add one
            PublicExperiments.objects.create(experiment=experiment)
        else:
            for p in pe:
                p.delete()
        return summary(request, experiment_id)


def delete_experiment(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    permission = check_user(request, experiment)
    if not permission == 'edit':
        return HttpResponse("You don't have the permission to do this!")
    else:
        experiment.delete()

    # different return page for staff
    if request.user.is_superuser:
        return list_all_experiments(request)

    # normally return to the basicviz index page
    return basicviz_index(request)
