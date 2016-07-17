from django.shortcuts import render
from django.http import HttpResponse

from basicviz.models import Experiment,Document,FeatureInstance,DocumentMass2Motif,FeatureMass2MotifInstance

def index(request):
	return HttpResponse('Hello')


def show_docs(request,experiment_id):
	experiment = Experiment.objects.get(id = experiment_id)
	documents = Document.objects.filter(experiment = experiment)
	context_dict = {}
	context_dict['experiment'] = experiment
	context_dict['documents'] = documents
	return render(request,'basicviz/show_docs.html',context_dict)

def show_doc(request,doc_id):
	document = Document.objects.get(id=doc_id)
	features = FeatureInstance.objects.filter(document = document)
	# features = sorted(features,key=lambda x:x.intensity,reverse=True)
	context_dict = {'document':document,'features':features}
	experiment = document.experiment
	context_dict['experiment'] = experiment
	mass2motif_instances = DocumentMass2Motif.objects.filter(document = document).order_by('-probability')
	context_dict['mass2motifs'] = mass2motif_instances
	feature_mass2motif_instances = {}
	for feature in features:
		feature_mass2motif_instances[feature] = FeatureMass2MotifInstance.objects.filter(document = document,featureinstance=feature)
	context_dict['fm2m'] = feature_mass2motif_instances

	a = FeatureMass2MotifInstance.objects.filter(document=document)[0]
	return render(request,'basicviz/show_doc.html',context_dict)