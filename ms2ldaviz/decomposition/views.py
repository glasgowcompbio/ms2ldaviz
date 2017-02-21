from django.shortcuts import render

from decomposition.models import GlobalMotif,DocumentGlobalMass2Motif
from basicviz.models import Mass2MotifInstance,Experiment
# Create your views here.
def view_parents(request,mass2motif_id,experiment_id):
	context_dict = {}
	experiment = Experiment.objects.get(id = experiment_id)
	mass2motif = GlobalMotif.objects.get(id = mass2motif_id)
	context_dict['mass2motif'] = mass2motif
	context_dict['experiment'] = experiment
	dm2ms = DocumentGlobalMass2Motif.objects.filter(mass2motif = mass2motif).order_by('-probability')

	originalfeatures = Mass2MotifInstance.objects.filter(mass2motif = mass2motif.originalmotif)

	context_dict['motif_features'] = originalfeatures
	context_dict['dm2ms'] = dm2ms
	return render(request, 'decomposition/view_parents.html',context_dict)