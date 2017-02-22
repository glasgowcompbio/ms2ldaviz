from django.shortcuts import render,HttpResponse

import json

from decomposition.models import GlobalMotif,DocumentGlobalMass2Motif
from basicviz.models import Mass2MotifInstance,Experiment

from options.views import get_option

from decomposition_functions import get_parents_decomposition
# Create your views here.
def view_parents(request,mass2motif_id,experiment_id):
    context_dict = {}
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motif = GlobalMotif.objects.get(id = mass2motif_id)
    context_dict['mass2motif'] = mass2motif
    context_dict['experiment'] = experiment


    edge_choice = get_option('default_doc_m2m_score',experiment = experiment)
    edge_thresh = get_option('doc_m2m_threshold',experiment = experiment)

    if edge_choice == 'probability':
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, probability__gte = edge_thresh).order_by('-probability')
    else:
        dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif, overlap_score__gte = edge_thresh).order_by('-overlap_score')

    originalfeatures = Mass2MotifInstance.objects.filter(mass2motif = mass2motif.originalmotif)

    context_dict['motif_features'] = originalfeatures
    context_dict['dm2ms'] = dm2ms
    return render(request, 'decomposition/view_parents.html',context_dict)

def get_parents(request,experiment_id,mass2motif_id):
    experiment = Experiment.objects.get(id = experiment_id)
    parent_data = get_parents_decomposition(mass2motif_id,vo_id = None,experiment = experiment)
    return HttpResponse(json.dumps(parent_data), content_type='application/json')

