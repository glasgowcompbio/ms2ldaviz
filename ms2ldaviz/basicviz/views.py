from django.shortcuts import render
from django.http import HttpResponse
import networkx as nx
from networkx.readwrite import json_graph
import json
import jsonpickle

from basicviz.models import Experiment,Document,FeatureInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Mass2Motif

def index(request):
    experiments = Experiment.objects.all()
    context_dict = {'experiments':experiments}
    return render(request,'basicviz/index.html',context_dict)


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
        feature_mass2motif_instances[feature] = FeatureMass2MotifInstance.objects.filter(featureinstance=feature)


    plot_fragments = []
    if len(features) > 0:
        for feature_instance in features:
            if feature_instance.feature.name.startswith('fragment'):
                mass = float(feature_instance.feature.name.split('_')[1])
                plot_fragments.append((mass,feature_instance.intensity))

    parent_peak = []
    metadata = jsonpickle.decode(document.metadata)
    if 'parentmass' in metadata:
        print metadata['parentmass']
        parent_peak.append((float(metadata['parentmass']),100))
    print parent_peak
    context_dict['fm2m'] = feature_mass2motif_instances
    context_dict['plot_fragments'] = plot_fragments
    context_dict['plot_parent'] = [parent_peak]
    return render(request,'basicviz/show_doc.html',context_dict)

def get_doc(request,doc_id):
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document = document)
    plot_fragments = []
    metadata = jsonpickle.decode(document.metadata)
    parent_mass = float(metadata['parentmass'])
    parent_data = (parent_mass,100.0)

    plot_fragments.append(parent_data)
    child_data = []
    max_intensity = 0.0
    if len(features) > 0:
        for feature_instance in features:
            if feature_instance.feature.name.startswith('fragment'):
                mass = float(feature_instance.feature.name.split('_')[1])
                if feature_instance.intensity > max_intensity:
                    max_intensity = feature_instance.intensity
                child_data.append((mass,mass,0,feature_instance.intensity,1,'red'))
            else:
                mass = float(feature_instance.feature.name.split('_')[1])
                if feature_instance.intensity > max_intensity:
                    max_intensity = feature_instance.intensity
                child_data.append((parent_mass-mass,parent_mass,feature_instance.intensity,feature_instance.intensity,0,'red'))
    child_data = [(m1,m2,i1*100.0/max_intensity,i2*100.0/max_intensity,t,c) for m1,m2,i1,i2,t,c in child_data]
    plot_fragments.append(child_data)

    return HttpResponse(json.dumps(plot_fragments),content_type='application/json')

def get_doc_topics(request,doc_id):
    colours = ['red','green','blue','black','yellow']
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document = document)
    plot_fragments = []
    metadata = jsonpickle.decode(document.metadata)
    parent_mass = float(metadata['parentmass'])
    parent_data = (parent_mass,100.0)

    plot_fragments.append(parent_data)
    child_data = []
    loss_data = []

    # Only colours the first five
    topics = sorted(DocumentMass2Motif.objects.filter(document=document),key=lambda x:x.probability,reverse=True)
    topics_to_plot = []
    for i in range(5):
        if i == len(topics):
            break
        topics_to_plot.append(topics[i].mass2motif)
    
    print topics_to_plot

    max_intensity = 0.0
    topic_colours = {}
    colour_pos = 0
    for feature_instance in features:
        if feature_instance.intensity > max_intensity:
            max_intensity = feature_instance.intensity
    if len(features) > 0:
        for feature_instance in features:
            phi_values = FeatureMass2MotifInstance.objects.filter(featureinstance = feature_instance)
            if feature_instance.feature.name.startswith('fragment'):
                mass = float(feature_instance.feature.name.split('_')[1])
                cum_pos = 0.0
                this_intensity = feature_instance.intensity*100.0/max_intensity
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = phi_value.probability*this_intensity
                        if phi_value.mass2motif in topic_colours:
                            colour = topic_colours[phi_value.mass2motif]
                        else:
                            topic_colours[phi_value.mass2motif] = colours[colour_pos]
                            colour_pos += 1
                            colour = topic_colours[phi_value.mass2motif]
                        child_data.append((mass,mass,cum_pos,cum_pos + proportion,1,colour))
                        cum_pos += proportion
            else:
                mass = float(feature_instance.feature.name.split('_')[1])
                cum_pos = parent_mass - mass
                this_intensity = feature_instance.intensity*100.0/max_intensity
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = mass * phi_value.probability
                        if phi_value.mass2motif in topic_colours:
                            colour = topic_colours[phi_value.mass2motif]
                        else:
                            topic_colours[phi_value.mass2motif] = colours[colour_pos]
                            colour_pos += 1
                            colour = topic_colours[phi_value.mass2motif]
                        child_data.append((cum_pos,cum_pos+proportion,this_intensity,this_intensity,0,colour))
                        cum_pos += proportion
    plot_fragments.append(child_data)
    return HttpResponse(json.dumps(plot_fragments),content_type='application/json')



def start_viz(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment':experiment}

    # G = make_graph(experiment)
    # d = json_graph.node_link_data(G) 
    # context_dict = {'graph':d}
    # json.dump(d, open('/Users/simon/git/ms2ldaviz/ms2ldaviz/static/graph.json','w'),indent=2)
    return render(request,'basicviz/graph.html',context_dict)

def get_graph(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    G = make_graph(experiment)
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type='application/json')


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
    print topics
    for topic in topics:
        G.add_node(topic.name,group=2,name=topic.name,
            size=topic_scale_factor * topics[topic],
            special = False, in_degree = topics[topic],
            score = 1)

    documents = Document.objects.filter(experiment = experiment)
    for document in documents:
        # name = document.name
 #        name = self.lda_dict['doc_metadata'][doc].get('compound',doc)
        metadata = jsonpickle.decode(document.metadata)
        if 'compound' in metadata:
          name = metadata['compound']
        else:
          name = document.name
        G.add_node(name,group=1,name = name,size=20,
            type='square',peakid = document.name,special=False,
            in_degree=0,score=0)
        for docm2m in DocumentMass2Motif.objects.filter(document=document):
            if docm2m.mass2motif in topics and docm2m.probability > edge_thresh:
                G.add_edge(docm2m.mass2motif.name,document.name,weight = edge_scale_factor*docm2m.probability)
    return G
    # pass