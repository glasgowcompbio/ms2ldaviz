from django.shortcuts import render,HttpResponse
from basicviz.models import Experiment
import json
from term_classifier.models import SubClassifier
from term_classifier.classification_functions import predict
from django.views.decorators.csrf import csrf_exempt
# Create your views here.



@csrf_exempt
def classify_docs(request,experiment_id):
	e = Experiment.objects.get(id = experiment_id)
	classifiers = SubClassifier.objects.filter(experiment = e)
	
	docs = json.loads(request.POST['docs'])

	output = {}
	# output['n_classifiers'] = len(classifiers)
	for classifier in classifiers:
		output[classifier.term.name] = predict(classifier,docs)
	# output = docs

	return HttpResponse(json.dumps(output), content_type='application/json')
