import json
import math

import jsonpickle
import networkx as nx
import numpy as np
from django.http import HttpResponse, Http404
from django.shortcuts import render
from networkx.readwrite import json_graph
from scipy.stats import ttest_ind

from basicviz.forms import Mass2MotifMetadataForm, AlphaCorrelationForm, AlphaDEForm
from basicviz.models import Experiment, Mass2Motif, Mass2MotifInstance, MultiFileExperiment, MultiLink, Alpha, \
    AlphaCorrOptions
from massbank.views import get_massbank_form
from options.views import get_option
from views_index import index
from views_lda_single import get_docm2m


def get_individual_names(request, mf_id):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links]
    individual_names = [i.name for i in individuals]
    return HttpResponse(json.dumps(individual_names), content_type='application/json')


def get_alpha_matrix(request, mf_id):
    if True:
        mfe = MultiFileExperiment.objects.get(id=mf_id)

        if not mfe.alpha_matrix:

            links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
            individuals = [l.experiment for l in links]
            motifs = Mass2Motif.objects.filter(experiment=individuals[0])

            alp_vals = []
            for individual in individuals:
                motifs = individual.mass2motif_set.all().order_by('name')
                alp_vals.append([m.alpha_set.all()[0].value for m in motifs])

            alp_vals = map(list, zip(*alp_vals))
            alp_vals = [[motifs[i].name, motifs[i].annotation] + av + [float((np.array(av) / sum(av)).var())] for i, av
                        in enumerate(alp_vals)]

            data = json.dumps(alp_vals)
            mfe.alpha_matrix = jsonpickle.encode(alp_vals)
            mfe.save()
        else:
            alp_vals = jsonpickle.decode(mfe.alpha_matrix)
            data = json.dumps(alp_vals)

        return HttpResponse(data, content_type='application/json')
    else:
        raise Http404


def get_degree_matrix(request, mf_id):
    if request.is_ajax():
        mfe = MultiFileExperiment.objects.get(id=mf_id)
        if not mfe.degree_matrix:
            links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
            individuals = [l.experiment for l in links]
            deg_vals = []

            for individual in individuals:

                doc_m2m_threshold = get_option('doc_m2m_threshold', experiment=individual)
                if doc_m2m_threshold:
                    doc_m2m_threshold = float(doc_m2m_threshold)
                else:
                    doc_m2m_threshold = 0.0
                default_score = get_option('default_doc_m2m_score', experiment=individual)
                if not default_score:
                    default_score = 'probability'

                new_row = []
                motif_set = individual.mass2motif_set.all().order_by('name')
                for motif in motif_set:
                    dm2m = motif.documentmass2motif_set.all()
                    if default_score == 'probability':
                        new_row.append(len([d for d in dm2m if d.probability > doc_m2m_threshold]))
                    else:
                        new_row.append(len([d for d in dm2m if d.overlap_score > doc_m2m_threshold]))
                deg_vals.append(new_row)

            deg_vals = map(list, zip(*deg_vals))
            deg_vals = [[motif_set[i].name, motif_set[i].annotation] + dv for i, dv in enumerate(deg_vals)]

            data = json.dumps(deg_vals)
            mfe.degree_matrix = jsonpickle.encode(deg_vals)
            mfe.save()
        else:
            deg_vals = jsonpickle.decode(mfe.degree_matrix)
            data = json.dumps(deg_vals)
        return HttpResponse(data, content_type='application/json')
    else:
        raise Http404


def make_alpha_matrix(individuals, normalise=True):
    print "Creating alpha matrix"
    alp_vals = []
    for individual in individuals:
        motifs = individual.mass2motif_set.all().order_by('name')
        alp_vals.append([m.alpha_set.all()[0].value for m in motifs])

    alp_vals = map(list, zip(*alp_vals))
    new_alp_vals = []
    if normalise:
        for av in alp_vals:
            s = sum(av)
            nav = [a / s for a in av]
            new_alp_vals.append(nav)
        alp_vals = new_alp_vals

    return alp_vals


def wipe_cache(request, mf_id):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    mfe.alpha_matrix = None
    mfe.degree_matrix = None
    mfe.save()
    return index(request)


def get_doc_table(request, mf_id, motif_name):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links]

    individual_motifs = {}
    for individual in individuals:
        thismotif = Mass2Motif.objects.get(experiment=individual, name=motif_name)
        individual_motifs[individual] = thismotif

    doc_table = []
    individual_names = []
    peaksets = {}
    peakset_list = []
    peakset_masses = []
    for i, individual in enumerate(individuals):
        individual_names.append(individual.name)
        docs = get_docm2m(individual_motifs[individual])
        for doc in docs:
            peakset_index = -1
            ii = doc.document.intensityinstance_set.all()
            if len(ii) > 0:
                ii = ii[0]
                ps = ii.peakset
                if not ps in peaksets:
                    peaksets[ps] = {}
                    peakset_list.append(ps)
                    peakset_masses.append(ps.mz)
                peakset_index = peakset_list.index(ps)
                peaksets[ps][individual] = ii.intensity

            mz = 0
            rt = 0
            md = jsonpickle.decode(doc.document.metadata)
            if 'parentmass' in md:
                mz = md['parentmass']
            elif 'mz' in md:
                mz = md['mz']
            elif '_' in doc.document.name:
                split_name = doc.document.name.split('_')
                mz = float(split_name[0])
            if 'rt' in md:
                rt = md['rt']
            elif '_' in doc.document.name:
                split_name = doc.document.name.split('_')
                rt = float(split_name[1])

            doc_table.append([rt, mz, i, doc.probability, peakset_index])

    # Add the peaks to the peakset object that are not linked to a document
    # (i.e. the MS1 peak is present, but it wasn't fragmented)
    for ps in peaksets:
        # Grab the intensity instances for this peakset
        intensity_instances = ps.intensityinstance_set.all()
        # Extract the individual experiments that are represented
        individuals_present = [i.experiment for i in intensity_instances]
        # Loop over the experiment
        for individual in individuals:
            # If the experiment is not in the current peakset but there is an intensity instance
            if (not individual in peaksets[ps]) and individual in individuals_present:
                # Find the intensity instance
                int_int = filter(lambda x: x.experiment == individual, intensity_instances)
                peaksets[ps][individual] = int_int[0].intensity
                print ps, individual, int_int[0].intensity

    intensity_table = []
    unnormalised_intensity_table = []
    counts = []
    final_peaksets = []

    min_count_options = get_option('heatmap_minimum_display_count',experiment = individuals[0])
    # min_count_options = SystemOptions.objects.filter(key='heatmap_minimum_display_count')
    if len(min_count_options) > 0:
        min_count = int(min_count_options)
    else:
        min_count = 5

    log_intensities_options = get_option('log_peakset_intensities',experiment = individuals[0])
    # log_intensities_options = SystemOptions.objects.filter(key='log_peakset_intensities')
    if len(log_intensities_options) > 0:
        val = log_intensities_options
        if val == 'true':
            log_peakset_intensities = True
        else:
            log_peakset_intensities = False
    else:
        log_peakset_intensities = True
    
    normalise_heatmap_options = get_option('heatmap_normalisation',experiment = individuals[0])
    if len(normalise_heatmap_options) == 0:
        normalise_heatmap_options = 'none'

    for peakset in peaksets:
        new_row = []
        for individual in individuals:
            new_row.append(peaksets[peakset].get(individual, 0))
        count = sum([1 for i in new_row if i > 0])
        if min_count >= 0:
            nz_vals = [v for v in new_row if v > 0]
            if log_peakset_intensities:
                nz_vals = [np.log(v) for v in nz_vals]
                new_row = [np.log(v) if v > 0 else 0 for v in new_row]
            me = sum(nz_vals) / (1.0 * len(nz_vals))
            va = sum([v ** 2 for v in nz_vals]) / len(nz_vals) - me ** 2
            va = math.sqrt(va)
            maxval = max(nz_vals)

            if normalise_heatmap_options == 'none':
                intensity_table.append(new_row)
                unnormalised_intensity_table.append(new_row)
                counts.append(count)
                final_peaksets.append(peakset)
            elif normalise_heatmap_options == 'max':
                new_row_n = [v/maxval for v in new_row]
                intensity_table.append(new_row_n)
                unnormalised_intensity_table.append(new_row)
                counts.append(count)
                final_peaksets.append(peakset)
            elif normalise_heatmap_options == 'standard' and va > 0:
                # if variance is zero, skip...
                unnormalised_intensity_table.append(new_row)
                new_row_n = [(v - me) / va if v > 0 else 0 for v in new_row]
                intensity_table.append(new_row_n)
                counts.append(count)
                final_peaksets.append(peakset)

    # Order so that the most popular are at the top
    if len(final_peaksets) > 0:
        temp = zip(counts, intensity_table, final_peaksets)
        temp = sorted(temp, key=lambda x: x[0], reverse=True)
        counts, intensity_table, final_peaksets = zip(*temp)
        intensity_table = list(intensity_table)

    # Change the indexes in the doc table to match the new ordering
    for row in doc_table:
        old_ps_index = row[-1]
        if old_ps_index > -1:
            old_ps = peakset_list[old_ps_index]
            if old_ps in final_peaksets:
                new_ps_index = final_peaksets.index(old_ps)
            else:
                new_ps_index = -1
            row[-1] = new_ps_index

    final_peakset_masses = [p.mz for p in final_peaksets]
    final_peakset_rt = [p.rt for p in final_peaksets]

    return HttpResponse(json.dumps((
        individual_names, doc_table, intensity_table, final_peakset_masses, final_peakset_rt,
        unnormalised_intensity_table)), content_type='application/json')


def view_multi_m2m(request, mf_id, motif_name):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']
    context_dict = {'mfe': mfe}
    context_dict['motif_name'] = motif_name

    # get the features
    firstm2m = Mass2Motif.objects.get(name=motif_name, experiment=individuals[0])
    m2mfeatures = Mass2MotifInstance.objects.filter(mass2motif=firstm2m)
    m2mfeatures = sorted(m2mfeatures, key=lambda x: x.probability, reverse=True)
    context_dict['m2m_features'] = m2mfeatures

    individual_motifs = {}
    for individual in individuals:
        thism2m = Mass2Motif.objects.get(name=motif_name, experiment=individual)
        individual_motifs[individual] = thism2m

    context_dict['status'] = 'Edit metadata...'
    if request.method == 'POST':
        form = Mass2MotifMetadataForm(request.POST)
        if form.is_valid():
            new_annotation = form.cleaned_data['metadata']
            new_short_annotation = form.cleaned_data['short_annotation']
            for individual in individual_motifs:
                motif = individual_motifs[individual]
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

    firstm2m = Mass2Motif.objects.get(name=motif_name, experiment=individuals[0])
    metadata_form = Mass2MotifMetadataForm(
        initial={'metadata': firstm2m.annotation, 'short_annotation': firstm2m.short_annotation})
    context_dict['metadata_form'] = metadata_form

    massbank_form = get_massbank_form(firstm2m, m2mfeatures, mf_id=mf_id)
    context_dict['massbank_form'] = massbank_form

    # Get the m2m in the individual models
    individual_m2m = []
    alps = []
    doc_table = []
    individual_names = []
    peaksets = {}
    peakset_list = []
    peakset_masses = []
    for i, individual in enumerate(individuals):
        alpha = Alpha.objects.get(mass2motif=individual_motifs[individual])
        docs = get_docm2m(individual_motifs[individual])
        individual_m2m.append([individual, individual_motifs[individual], alpha, len(docs)])
        alps.append(alpha.value)

    # Compute the mean and variance
    tot_alps = sum(alps)
    m_alp = sum(alps) / len(alps)
    m_alp2 = sum([a ** 2 for a in alps]) / len(alps)
    var = m_alp2 - m_alp ** 2
    context_dict['alpha_variance'] = var
    context_dict['alphas'] = zip([i.name for i in individuals], alps)
    context_dict['individual_m2m'] = individual_m2m
    return render(request, 'basicviz/view_multi_m2m.html', context_dict)


def get_alphas(request, mf_id, motif_name):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links]
    alps = []
    for individual in individuals:
        m2m = Mass2Motif.objects.get(name=motif_name, experiment=individual)
        alpha = Alpha.objects.get(mass2motif=m2m)
        alps.append(alpha.value)

    alps = [[individuals[i].name, a] for i, a in enumerate(alps)]
    json_alps = json.dumps(alps)
    return HttpResponse(json_alps, content_type='application/json')


def get_degrees(request, mf_id, motif_name):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links]
    degs = []
    for individual in individuals:
        m2m = Mass2Motif.objects.get(name=motif_name, experiment=individual)
        docs = get_docm2m(m2m)
        degs.append(len(docs))

    degs = zip([i.name for i in individuals], degs)
    json_degs = json.dumps(degs)
    return HttpResponse(json_degs, content_type='application/json')


def alpha_pca(request, mf_id):
    # Returns a json object to be rendered into a pca plot
    # PCA is pre-computed
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    if mfe.pca:
        pca_data = jsonpickle.decode(mfe.pca)
    else:
        pca_data = []
    return HttpResponse(json.dumps(pca_data), content_type='application/json')


def multi_alphas(request, mf_id):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    context_dict = {'mfe': mfe}
    links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
    individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']
    context_dict['individuals'] = individuals

    alp_vals = []
    degrees = []
    context_dict['alp_vals'] = alp_vals
    context_dict['degrees'] = degrees
    context_dict['url'] = '/basicviz/alpha_pca/{}/'.format(mfe.id)
    return render(request, 'basicviz/multi_alphas.html', context_dict)


def alpha_correlation(request, mf_id):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    context_dict = {}
    context_dict['mfe'] = mfe

    if request.method == 'POST':
        form = AlphaCorrelationForm(request.POST)
        if form.is_valid():
            distance_score = form.cleaned_data['distance_score']
            edge_thresh = form.cleaned_data['edge_thresh']
            normalise_alphas = form.cleaned_data['normalise_alphas']
            max_edges = form.cleaned_data['max_edges']
            just_annotated = form.cleaned_data['just_annotated']
            acviz = AlphaCorrOptions.objects.get_or_create(multifileexperiment=mfe,
                                                           distance_score=distance_score,
                                                           edge_thresh=edge_thresh,
                                                           normalise_alphas=normalise_alphas,
                                                           max_edges=max_edges,
                                                           just_annotated=just_annotated)[0]
            context_dict['acviz'] = acviz
        else:
            context_dict['form'] = form
    else:
        context_dict['form'] = AlphaCorrelationForm()

    return render(request, 'basicviz/alpha_correlation.html', context_dict)


def get_alpha_correlation_graph(request, acviz_id):
    from itertools import combinations
    acviz = AlphaCorrOptions.objects.get(id=acviz_id)
    mfe = acviz.multifileexperiment
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    an_experiment = links[0].experiment
    motifs = Mass2Motif.objects.filter(experiment=an_experiment).order_by('name')

    if mfe.alpha_matrix:
        alp_vals_with_names = jsonpickle.decode(mfe.alpha_matrix)
        alp_vals = []
        for av in alp_vals_with_names:
            newav = av[2:-1]
            alp_vals.append(newav)

    else:
        alp_vals = make_alpha_matrix(individuals, normalise=True)

    motif_index = []
    an_motifs = []
    if acviz.just_annotated:
        for i, motif in enumerate(motifs):
            if 'annotation' in jsonpickle.decode(motif.metadata):
                an_motifs.append(motif)
                motif_index.append(i)
        motifs = an_motifs
    else:
        motif_index = range(len(motifs))

    # Add motifs as nodes
    G = nx.Graph()
    motif_names = []
    for motif in motifs:
        md = jsonpickle.decode(motif.metadata)
        display_name = md.get('annotation', motif.name)

        motif_names.append(motif.name)
        if 'annotation' in md:
            G.add_node(motif.name, name=display_name, col='#FF0000')
        else:
            G.add_node(motif.name, name=display_name, col='#333333')

    # add edges where the score is > thresh
    scores = []
    for i, j in combinations(range(len(motifs)), 2):
        a1 = np.array(alp_vals[motif_index[i]])
        a2 = np.array(alp_vals[motif_index[j]])

        if acviz.normalise_alphas:
            a1n = a1 / np.linalg.norm(a1)
            a2n = a2 / np.linalg.norm(a2)
        else:
            a1n = a1
            a2n = a2

        if acviz.distance_score == 'cosine':
            score = np.dot(a1n, a2n)
        elif acviz.distance_score == 'euclidean':
            score = np.sqrt((a1n - a2n) ** 2)
        elif acviz.distance_score == 'rms':
            score = np.sqrt(((a1n - a2n) ** 2).mean())
        elif acviz.distance_score == 'pearson':
            score = ((a1n - a1n.mean()) * (a2n - a2n.mean())).mean() / (a1n.std() * a2n.std())

        scores.append((i, j, score))

    if acviz.distance_score == 'cosine' or acviz.distance_score == 'pearson':
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
    else:
        scores = sorted(scores, key=lambda x: x[2])

    pos = 0
    while True:
        i, j, score = scores[pos]
        if (acviz.distance_score == 'cosine' or acviz.distance_score == 'pearson') and score > acviz.edge_thresh:
            G.add_edge(motif_names[i], motif_names[j], weight=score)
        elif (acviz.distance_score == 'euclidean' or acviz.distance_score == 'rms') and score > acviz.edge_thresh:
            G.add_edge(motif_names[i], motif_names[j])
        else:
            break
        pos += 1
        if pos > acviz.max_edges:
            break
        if pos >= len(scores):
            break

    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d), content_type='application/json')


def alpha_de(request, mfe_id):
    mfe = MultiFileExperiment.objects.get(id=mfe_id)
    context_dict = {'mfe': mfe}
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    tu = zip(individuals, individuals)
    tu = sorted(tu, key=lambda x: x[0].name)
    if request.method == 'POST':
        form = AlphaDEForm(tu, request.POST)
        if form.is_valid():
            group1_experiments = form.cleaned_data['group1']
            group2_experiments = form.cleaned_data['group2']
            motifs = individuals[0].mass2motif_set.all().order_by('name')

            if mfe.alpha_matrix:
                alp_vals_with_names = jsonpickle.decode(mfe.alpha_matrix)
                alp_vals = []
                for av in alp_vals_with_names:
                    newav = av[2:-1]
                    alp_vals.append(newav)

            else:
                alp_vals = make_alpha_matrix(individuals, normalise=True)

            group1_index = []
            group2_index = []
            motif_scores = []
            for experiment_name in group1_experiments:
                experiment = Experiment.objects.get(name=experiment_name)
                group1_index.append(individuals.index(experiment))
            for experiment_name in group2_experiments:
                experiment = Experiment.objects.get(name=experiment_name)
                group2_index.append(individuals.index(experiment))
            for i, alp in enumerate(alp_vals):
                a = np.array(alp)
                g1 = a[group1_index]
                g2 = a[group2_index]
                de = (g1.mean() - g2.mean()) / (g1.std() + g2.std())
                t, p = ttest_ind(g1, g2, equal_var=False)
                motif_scores.append((motifs[i], de, p))
            context_dict['motif_scores'] = motif_scores



        else:
            context_dict['alpha_de_form'] = form
    else:
        form = AlphaDEForm(tu)
        context_dict['alpha_de_form'] = form
    return render(request, 'basicviz/alpha_de.html', context_dict)


def get_multifile_mass2motif_metadata(request, mf_id, motif_name):
    mfe = MultiFileExperiment.objects.get(id=mf_id)
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    first_experiment = individuals[0]
    mass2motif = Mass2Motif.objects.get(experiment=first_experiment, name=motif_name)
    md = jsonpickle.decode(mass2motif.metadata)
    return HttpResponse(json.dumps(md), content_type='application/json')
