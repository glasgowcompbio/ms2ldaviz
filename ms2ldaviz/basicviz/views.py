from django.shortcuts import render
from django.http import HttpResponse
import networkx as nx
from networkx.readwrite import json_graph

from basicviz.models import Experiment,Document,FeatureInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Mass2Motif

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

def start_viz(request,experiment_id):
	experiment = Experiment.objects.get(id=experiment_id)
	context_dict = {}

	G = make_graph(experiment)
	d = json_graph.node_link_data(G) 

	context_dict = {'graph':d}
	return render(request,'basicviz/show_viz.html',context_dict)

def make_graph(experiment,edge_thresh = 0.05,min_degree = 10,topic_scale_factor = 5,edge_scale_factor=5):
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    # Find the degrees
    topics = {}
    for mass2motif in mass2motifs:
    	topics[mass2motif] = 0
        docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif)
        for d in docm2ms:
        	if d.probability > edge_thresh:
        		topics[mass2motif] += 1
    to_remove = []
    for topic in topics:
        if topics[topic] < min_degree:
            to_remove.append(topic)
    for topic in to_remove:
        del topics[topic]

    print "Found {} topics".format(len(topics))

    # Add the topics to the graph
    G = nx.Graph()
    for topic in topics:
        G.add_node(topic.name,group=2,name=topic.name,
            size=topic_scale_factor * topics[topic],
            special = False, in_degree = topics[topic],
            score = 1)

 #    # Add the nodes to the graph
 #    for doc in self.lda_dict['theta']:
 #        # Get the compound as the name if it exists
 #        name = self.lda_dict['doc_metadata'][doc].get('compound',doc)
 #        G.add_node(doc,group=1,name = name,size=20,
 #            type='square',peakid = name,special=False,
 #            in_degree=0,score=0)
 #        for topic in self.lda_dict['theta'][doc]:
 #            G.add_edge(topic,doc,weight = edge_scale_factor*self.lda_dict['theta'][doc][topic])
 	return G
	# pass