from django.shortcuts import render,HttpResponse

import json

from decomposition.models import GlobalMotif,DocumentGlobalMass2Motif,Decomposition,GlobalMotifsToSets
from decomposition.forms import DecompVizForm
from basicviz.models import Mass2MotifInstance,Experiment,Document

from options.views import get_option

from decomposition_functions import get_parents_decomposition,get_decomp_doc_context_dict,get_parent_for_plot_decomp,make_word_graph,make_intensity_graph,make_decomposition_graph
from basicviz.views import views_lda_single
# Create your views here.
def view_parents(request,mass2motif_id,decomposition_id):
    context_dict = {}
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    mass2motif = GlobalMotif.objects.get(id = mass2motif_id)
    context_dict['mass2motif'] = mass2motif
    context_dict['experiment'] = experiment
    context_dict['decomposition'] = decomposition

    # Thought -- should these options be decomposition specific?
    edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)

    if edge_choice == 'probability':
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, probability__gte = edge_thresh,decomposition = decomposition).order_by('-probability')
    elif edge_choice == 'overlap_score':
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, overlap_score__gte = edge_thresh, decomposition = decomposition).order_by('-overlap_score')
    elif edge_choice == 'both':
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, overlap_score__gte = edge_thresh, probability__gte = edge_thresh,decomposition = decomposition).order_by('-overlap_score')
    else:
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, probability__gte = edge_thresh,decomposition = decomposition).order_by('-probability')

    originalfeatures = Mass2MotifInstance.objects.filter(mass2motif = mass2motif.originalmotif)

    context_dict['motif_features'] = originalfeatures
    context_dict['dm2ms'] = dm2ms
    return render(request, 'decomposition/view_parents.html',context_dict)

def get_parents(request,decomposition_id,mass2motif_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    parent_data = get_parents_decomposition(mass2motif_id,decomposition,vo_id = None,experiment = experiment)
    return HttpResponse(json.dumps(parent_data), content_type='application/json')

def get_word_graph(request,mass2motif_id,vo_id,decomposition_id):
    data_for_json = make_word_graph(request,mass2motif_id,vo_id,decomposition_id)
    return HttpResponse(json.dumps(data_for_json),content_type='application/json')

def get_intensity_graph(request,mass2motif_id,vo_id,decomposition_id):
    data_for_json = make_intensity_graph(request,mass2motif_id,vo_id,decomposition_id)
    return HttpResponse(json.dumps(data_for_json),content_type='application/json')

def show_parents(request,decomposition_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    context_dict = {}
    context_dict['decomposition'] = decomposition
    context_dict['experiment'] = experiment
    
    # Get the documents
    documents = Document.objects.filter(experiment = experiment)
    context_dict['documents'] = documents

    return render(request,'decomposition/show_parents.html',context_dict)

def show_doc(request,decomposition_id,doc_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    document = Document.objects.get(id = doc_id)
    context_dict = get_decomp_doc_context_dict(decomposition,document)
    
    context_dict['decomposition'] = decomposition
    context_dict['experiment'] = experiment
    context_dict['document'] = document

    if document.csid:
        context_dict['csid'] = document.csid
        
    if document.image_url:
        context_dict['image_url'] = document.image_url

    return render(request,'decomposition/show_doc.html',context_dict) 

def show_motifs(request,decomposition_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    motifset = decomposition.motifset
    motiftoset = GlobalMotifsToSets.objects.filter(motifset = motifset)
    mass2motifs = [m.motif for m in motiftoset]

    context_dict = {}
    context_dict['decomposition'] = decomposition
    context_dict['experiment'] = experiment
    context_dict['mass2motifs'] = mass2motifs
   
    return render(request,'decomposition/view_mass2motifs.html',context_dict)

def get_doc_topics(request, decomposition_id,doc_id):
    document = Document.objects.get(id = doc_id)
    decomposition = Decomposition.objects.get(id = decomposition_id)
    score_type = get_option('default_doc_m2m_score',experiment = document.experiment)
    plot_fragments = [get_parent_for_plot_decomp(decomposition,document,edge_choice=score_type,get_key = True)]
    return HttpResponse(json.dumps(plot_fragments), content_type='application/json')

def start_viz(request,decomposition_id):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    context_dict = {}
    context_dict['decomposition'] = decomposition
    context_dict['experiment'] = experiment

    # add form stuff here!
    if request.method == 'POST':
        form = DecompVizForm(request.POST)
        if form.is_valid():
            min_degree = form.cleaned_data['min_degree']
        context_dict['vo'] = {'min_degree':min_degree,'random_seed':'hello'}
        return render(request,'decomposition/graph.html',context_dict)    
    else:
        context_dict['viz_form'] = DecompVizForm()

    return render(request,'decomposition/viz_form.html',context_dict)


    

def get_graph(request,decomposition_id,min_degree):
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    min_degree = int(min_degree)

    edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    edge_thresh = float(get_option('doc_m2m_threshold',experiment = experiment))

    d = make_decomposition_graph(decomposition,experiment,min_degree = min_degree,edge_thresh = edge_thresh,
                                edge_choice = edge_choice,topic_scale_factor = 5, edge_scale_factor = 5)
    return HttpResponse(json.dumps(d),content_type = 'application/json')

