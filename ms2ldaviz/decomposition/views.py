from django.shortcuts import render,HttpResponse
from ipware.ip import get_ip
import json

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from decomposition.models import GlobalMotif,DocumentGlobalMass2Motif,Decomposition,GlobalMotifsToSets,MotifSet,FeatureSet,APIBatchResult
from decomposition.forms import DecompVizForm,NewDecompositionForm,BatchDecompositionForm,SpectrumForm,MotifsetAnnotationForm
from basicviz.models import Mass2MotifInstance,Experiment,Document,JobLog,VizOptions
from basicviz.forms import VizForm
from options.views import get_option

from ms1analysis.models import DecompositionAnalysis

from decomposition_functions import get_parents_decomposition,get_decomp_doc_context_dict,get_parent_for_plot_decomp,make_word_graph,make_intensity_graph,make_decomposition_graph,parse_spectrum_string
from decomposition.tasks import api_batch_task
from basicviz.views import views_lda_single

from uploads.tasks import just_decompose_task

from basicviz.constants import EXPERIMENT_STATUS_CODE

from networkx.readwrite import json_graph


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
    
    ready, _ = EXPERIMENT_STATUS_CODE[1]
    choices = [(analysis.id, analysis.name + '(' + analysis.description + ')') for analysis in DecompositionAnalysis.objects.filter(decomposition=decomposition, status=ready)]


    # add form stuff here!
    if request.method == 'POST':
        # form = DecompVizForm(request.POST)
        viz_form = VizForm(choices,request.POST)
        if viz_form.is_valid():
            min_degree = viz_form.cleaned_data['min_degree']
        if len(viz_form.cleaned_data['ms1_analysis']) == 0 or viz_form.cleaned_data['ms1_analysis'][0] == '':
            ms1_analysis_id = None
        else:
            ms1_analysis_id = viz_form.cleaned_data['ms1_analysis'][0]
        vo = VizOptions.objects.get_or_create(experiment=experiment,
                                                  min_degree=min_degree,
                                                  ms1_analysis_id=ms1_analysis_id)[0]
        context_dict['vo'] = vo
        return render(request,'decomposition/graph.html',context_dict)    
    else:
        # context_dict['viz_form'] = DecompVizForm()
        context_dict['viz_form'] = VizForm(choices)

    return render(request,'decomposition/viz_form.html',context_dict)


    

def get_graph(request,decomposition_id,vo_id):
    vo = VizOptions.objects.get(id = vo_id)
    decomposition = Decomposition.objects.get(id = decomposition_id)
    experiment = decomposition.experiment
    min_degree = int(vo.min_degree)

    # edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    # edge_thresh = float(get_option('doc_m2m_threshold',experiment = experiment))

    G = make_decomposition_graph(decomposition,experiment,min_degree = min_degree,
                                ms1_analysis_id = vo.ms1_analysis_id)
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type = 'application/json')


def new_decomposition(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {}
    context_dict['experiment'] = experiment
    if request.method == 'POST':
        form = NewDecompositionForm(request.POST)
        if form.is_valid():
            decomposition,status = Decomposition.objects.get_or_create(experiment = experiment,
                                                    motifset = form.cleaned_data['motifset'],
                                                    name = form.cleaned_data['name'])
            JobLog.objects.create(user = request.user,experiment = experiment,tasktype = 'Decompose with ' + str(form.cleaned_data['motifset']))
            just_decompose_task.delay(decomposition.id)
        else:
            context_dict['form'] = form
    else:
        form = NewDecompositionForm()
        context_dict['form'] = form
    return render(request,'decomposition/new_decomposition.html',context_dict)

@csrf_exempt
def get_motifset_annotations(request):
    json_data = {'status':'FAILED'}
    if request.method == 'POST':
        maform = MotifsetAnnotationForm(request.POST)
        if maform.is_valid():
            json_data = {}
            motifset_name = maform.cleaned_data['motifset']
            json_data['motifset'] = motifset_name
            try:
                motifset = MotifSet.objects.get(name = motifset_name)
            except:
                all_motifsets = MotifSet.objects.all()
                status_string = 'unknown motifset, try: ' + ' '.join([m.name for m in all_motifsets])
                json_data = {'status':status_string}
                return JsonResponse(json_data)

            global_motif_links = GlobalMotifsToSets.objects.filter(motifset = motifset)
            global_motifs = [g.motif for g in global_motif_links]
            original_motifs = [g.originalmotif for g in global_motifs]
            ma = []
            for o in original_motifs:
                if o.annotation:
                    ma.append([o.name,o.annotation])
            json_data['annotations'] = ma
            json_data['status'] = 'SUCCESS'
    return JsonResponse(json_data)




@csrf_exempt
def batch_decompose(request):
    json_data = {'status':'FAILED'}
    if request.method == 'POST':
        batchform = BatchDecompositionForm(request.POST)
        if batchform.is_valid():
            json_data = {}
            spectra = json.loads(batchform.cleaned_data['spectra'])
            n_spectra = len(spectra)
            json_data['n_spectra'] = n_spectra
            motifset_name = batchform.cleaned_data['motifset']
            json_data['motifset'] = motifset_name
            try:
                motifset = MotifSet.objects.get(name = motifset_name)
            except:
                all_motifsets = MotifSet.objects.all()
                status_string = 'unknown motifset, try: ' + ' '.join([m.name for m in all_motifsets])
                json_data = {'status':status_string}
                return JsonResponse(json_data)


            featureset = motifset.featureset
            json_data['featureset'] = featureset.name

            batch_result = APIBatchResult.objects.create(results = 'submitted')

            api_batch_task.delay(spectra,featureset.id,motifset.id,batch_result.id)

            send_ip = get_ip(request)
            job_string = "Batch decompose, {} spectra, motifset: {}, from {}".format(n_spectra,motifset_name,send_ip)
            JobLog.objects.create(tasktype = job_string)

            json_data['result_id'] = str(batch_result.id)
            json_data['result_url'] = 'http://ms2lda.org/decomposition/api/batch_results/{}/'.format(batch_result.id)
            # doc_dict = make_documents(spectra,featureset)
            # results = api_decomposition(doc_dict,motifset)

            # json_data['results'] = results

            json_data['status'] = 'SUBMITTED'


    return JsonResponse(json_data)

def batch_results(request,result_id):
    batch_result = APIBatchResult.objects.get(id = result_id)
    try:
        results = json.loads(batch_result.results)
    except:
        results = {'status':batch_result.results}
    return JsonResponse(results)

def decompose_spectrum(request):
    context_dict = {}
    if request.method == 'POST':
        form = SpectrumForm(request.POST)
        if form.is_valid():
            # Convert the spectrum into json
            motifset_id = int(form.cleaned_data['motifset'])
            motifset = MotifSet.objects.get(id = motifset_id)
            featureset = motifset.featureset
            peaks = parse_spectrum_string(form.cleaned_data['spectrum'])
            parentmass = float(form.cleaned_data['parentmass'])
            spectra = [('spectrum_{}'.format(parentmass),parentmass,peaks)]
            batch_result = APIBatchResult.objects.create(results = 'submitted')
            api_batch_task.delay(spectra,featureset.id,motifset.id,batch_result.id)
            result_url = 'http://ms2lda.org/decomposition/api/batch_results/{}/'.format(batch_result.id)
            context_dict['result_url'] = result_url
            context_dict['result_id'] = batch_result.id
            print peaks
            # context_dict['document'] = peaks
            return render(request,'decomposition/decompose_spectrum.html',context_dict)
    else:
        form = SpectrumForm()
    context_dict['spectrum_form'] = form
    return render(request,'decomposition/decompose_spectrum.html',context_dict)

def pretty_results(request,result_id):
    batch_result = APIBatchResult.objects.get(id = result_id)
    context_dict = {}
    context_dict['result'] = batch_result
    try:
        result_json = json.loads(batch_result.results)
        context_dict['alpha'] = result_json['alpha']
        motifset = MotifSet.objects.get(name = result_json['motifset'])
        context_dict['motifset'] = motifset
        doc_name = result_json['terms'].keys()[0]
        decomps = result_json['decompositions']

        all_global_motifs = GlobalMotifsToSets.objects.filter(motifset = motifset)

        global_motif_dict = {}
        for global_motif in all_global_motifs:
            global_motif_dict[global_motif.motif.name] = global_motif.motif

        decomp_list = []
        for doc_name,motif_info in decomps.items():
            doc_list = []
            for global_name,original_name,p,o,annotation in motif_info:
                global_motif = global_motif_dict[global_name]
                doc_list.append((global_motif,p,o))
            doc_list = sorted(doc_list,key = lambda x: x[1],reverse = True)
            decomp_list.append((doc_name,doc_list))
        context_dict['decomp_list'] = decomp_list

        term_list = []
        for doc_name,terms in result_json['terms'].items():
            doc_term_list = []
            for term_name,term_type,term_score in terms:
                doc_term_list.append((term_name,term_type,term_score))
            doc_term_list = sorted(doc_term_list,key = lambda x: x[2],reverse = True)
            term_list.append((doc_name,doc_term_list))
        context_dict['terms'] = term_list

        context_dict['finished'] = True
    except:
        context_dict['finished'] = False
    return render(request,'decomposition/pretty_results.html',context_dict)
