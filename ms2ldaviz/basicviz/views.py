from django.shortcuts import render
from django.http import HttpResponse
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
import json
import jsonpickle
import csv

from basicviz.models import Experiment,Document,FeatureInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Mass2Motif,Mass2MotifInstance

def index(request):
    experiments = Experiment.objects.all()
    context_dict = {'experiments':experiments}
    return render(request,'basicviz/basicviz.html',context_dict)


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
    feature_mass2motif_instances = []
    for feature in features:
        if feature.intensity > 0:
            feature_mass2motif_instances.append((feature,FeatureMass2MotifInstance.objects.filter(featureinstance=feature)))

    feature_mass2motif_instances = sorted(feature_mass2motif_instances,key = lambda x:x[0].intensity,reverse=True)

    if document.csid:
        context_dict['image_url'] = 'http://www.chemspider.com/ImagesHandler.ashx?id=' + str(document.csid)
        context_dict['csid'] = document.csid
    elif document.inchikey:
        from chemspipy import ChemSpider
        cs = ChemSpider('b07b7eb2-0ba7-40db-abc3-2a77a7544a3d') 
        results = cs.search(document.inchikey)
        if results:
            context_dict['image_url'] = results[0].image_url
            context_dict['csid'] = results[0].csid

    context_dict['fm2m'] = feature_mass2motif_instances
    return render(request,'basicviz/show_doc.html',context_dict)

def view_parents(request,motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    context_dict = {'mass2motif':motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif = motif).order_by('-probability')
    context_dict['motif_features'] = motif_features
    return render(request,'basicviz/view_parents.html',context_dict)

def mass2motif_feature(request,fm2m_id):
    feature_mass2motif = Mass2MotifInstance.objects.get(id = fm2m_id)
    context_dict['feature_mass2motif'] = feature_mass2motif
    return render(request,'basicviz/mass2motif_feature.html',context_dict)

def get_parents(request,motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif)
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        if dm.probability > 0.05:
            document = dm.document
            parent_data.append(get_doc_for_plot(document.id,motif_id))
    return HttpResponse(json.dumps(parent_data),content_type = 'application/json')

def get_annotated_parents(request,motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif)
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        if dm.probability > 0.05:
            document = dm.document
            if len(document.annotation) > 0:
                parent_data.append(get_doc_for_plot(document.id,motif_id))
    return HttpResponse(json.dumps(parent_data),content_type = 'application/json')


def view_mass2motifs(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    context_dict = {'mass2motifs':mass2motifs}
    context_dict['experiment'] = experiment
    return render(request,'basicviz/view_mass2motifs.html',context_dict)

def get_doc_for_plot(doc_id,motif_id = None):
    colours = ['red','green','black','yellow']
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document = document)
    plot_fragments = []

    # Get the parent info
    metadata = jsonpickle.decode(document.metadata)
    parent_mass = float(metadata['parentmass'])
    parent_data = (parent_mass,100.0,document.name,document.annotation)
    plot_fragments.append(parent_data)
    child_data = []

    # Only colours the first five
    if motif_id == None:
        topic_colours = {}
        topics = sorted(DocumentMass2Motif.objects.filter(document=document),key=lambda x:x.probability,reverse=True)
        topics_to_plot = []
        for i in range(4):
            if i == len(topics):
                break
            topics_to_plot.append(topics[i].mass2motif)
            topic_colours[topics[i].mass2motif] = colours[i]
    else:
        topic = Mass2Motif.objects.get(id = motif_id)
        topics_to_plot = [topic]
        topic_colours = {topic:'red'}

    max_intensity = 0.0
    for feature_instance in features:
        if feature_instance.intensity > max_intensity:
            max_intensity = feature_instance.intensity

    if len(features) > 0:
        for feature_instance in features:
            phi_values = FeatureMass2MotifInstance.objects.filter(featureinstance = feature_instance)
            mass = float(feature_instance.feature.name.split('_')[1])
            this_intensity = feature_instance.intensity*100.0/max_intensity
            feature_name = feature_instance.feature.name
            if feature_name.startswith('fragment'):
                cum_pos = 0.0
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = phi_value.probability*this_intensity
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append((mass,mass,cum_pos,cum_pos + proportion,1,colour,feature_name))
                        cum_pos += proportion
                    else:
                        proportion = phi_value.probability*this_intensity
                        other_topics += proportion
                child_data.append((mass,mass,this_intensity - other_topics,this_intensity,1,'gray',feature_name))
            else:
                cum_pos = parent_mass - mass
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = mass * phi_value.probability
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append((cum_pos,cum_pos+proportion,this_intensity,this_intensity,0,colour,feature_name))
                        cum_pos += proportion
                    else:
                        proportion = mass * phi_value.probability
                        other_topics += proportion
                child_data.append((parent_mass - other_topics,parent_mass,this_intensity,this_intensity,0,'gray',feature_name))
    plot_fragments.append(child_data)
    return plot_fragments


def get_doc_topics(request,doc_id):
    plot_fragments = get_doc_for_plot(doc_id)
    return HttpResponse(json.dumps(plot_fragments),content_type='application/json')


def start_viz(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment':experiment}
    initial_motif = Mass2Motif.objects.filter(experiment = experiment)[0]
    context_dict['initial_motif'] = initial_motif
    # G = make_graph(experiment)
    # d = json_graph.node_link_data(G) 
    # context_dict = {'graph':d}
    # json.dump(d, open('/Users/simon/git/ms2ldaviz/ms2ldaviz/static/graph.json','w'),indent=2)
    return render(request,'basicviz/graph.html',context_dict)

def start_annotated_viz(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment':experiment}
    # This is a bit of a hack to make sure that the initial motif is one in the graph
    documents = Document.objects.filter(experiment = experiment)
    for document in documents:
        if len(document.annotation) > 0:
            for docm2m in DocumentMass2Motif.objects.filter(document = document):
                if docm2m.probability > 0.05:
                    context_dict['initial_motif'] = docm2m.mass2motif
                    break
    # initial_motif = Mass2Motif.objects.filter(experiment = experiment)[0]
    # context_dict['initial_motif'] = initial_motif
    # G = make_graph(experiment)
    # d = json_graph.node_link_data(G) 
    # context_dict = {'graph':d}
    # json.dump(d, open('/Users/simon/git/ms2ldaviz/ms2ldaviz/static/graph.json','w'),indent=2)
    return render(request,'basicviz/annotated_graph.html',context_dict)


def get_graph(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    G = make_graph(experiment)
    # G = make_annotated_graph(experiment)
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type='application/json')

def get_annotated_graph(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    # G = make_graph(experiment)
    G = make_annotated_graph(experiment)
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type='application/json')



def make_annotated_graph(experiment,edge_thresh = 0.05,min_degree = 1,
    topic_scale_factor = 20,edge_scale_factor = 5):
    # makes the graph with just the documents that are annotated
    documents = Document.objects.filter(experiment = experiment)
    annotated_documents = []
    topic_degrees = {}
    G = nx.Graph()
    for document in documents:
        metadata = jsonpickle.decode(document.metadata)
        if 'annotation' in metadata:
            annotated_documents.append(document)
            G.add_node(document.name,group=1,name = metadata['annotation'],size=20,
                type='square',peakid = document.name,special=False,
                in_degree=0,score=0,is_topic = False)
            for docm2m in DocumentMass2Motif.objects.filter(document=document):
                if docm2m.probability > edge_thresh:
                    if docm2m.mass2motif in topic_degrees:
                        topic_degrees[docm2m.mass2motif] += 1
                    else:
                        topic_degrees[docm2m.mass2motif] = 1
    for topic in topic_degrees:
        if topic_degrees[topic] >= min_degree:
            metadata = jsonpickle.decode(topic.metadata)
            if 'annotation' in metadata:
                name = metadata['annotation']
                special = True
            else:
                name = topic.name
                special = False
            G.add_node(topic.name,group=2,name=name,size=topic_scale_factor * topic_degrees[topic],
                special = special, in_degree = topic_degrees[topic],
                score = 1,node_id = topic.id,is_topic = True)
            for docm2m in DocumentMass2Motif.objects.filter(mass2motif = topic):
                if docm2m.probability > edge_thresh and docm2m.document in annotated_documents:
                    G.add_edge(topic.name,docm2m.document.name,weight = edge_scale_factor*docm2m.probability)
    return G


def make_graph(experiment,edge_thresh = 0.05,min_degree = 5,
    topic_scale_factor = 5,edge_scale_factor=5):
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


    # Add the topics to the graph
    G = nx.Graph()
    for topic in topics:
        metadata = jsonpickle.decode(topic.metadata)
        if 'annotation' in metadata:
            G.add_node(topic.name,group=2,name=metadata['annotation'],
                size=topic_scale_factor * topics[topic],
                special = True, in_degree = topics[topic],
                score = 1,node_id = topic.id,is_topic = True)
        else:
            G.add_node(topic.name,group=2,name=topic.name,
                size=topic_scale_factor * topics[topic],
                special = False, in_degree = topics[topic],
                score = 1,node_id = topic.id,is_topic = True)

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
            in_degree=0,score=0,is_topic = False)
        for docm2m in DocumentMass2Motif.objects.filter(document=document):
            if docm2m.mass2motif in topics and docm2m.probability > edge_thresh:
                G.add_edge(docm2m.mass2motif.name,document.name,weight = edge_scale_factor*docm2m.probability)
    return G
    
def document_pca(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {}
    context_dict['experiment'] = experiment

    return render(request,'basicviz/document_pca.html',context_dict)

def get_pca_data(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    theta_data = []
    documents = Document.objects.filter(experiment = experiment)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    n_mass2motifs = len(mass2motifs)
    m2mindex = {}
    msmpos = 0
    for document in documents:
        new_theta = [0 for i in range(n_mass2motifs)]
        dm2ms = DocumentMass2Motif.objects.filter(document = document)
        for dm2m in dm2ms:
            if dm2m.mass2motif.name in m2mindex:
                m2mpos = m2mindex[dm2m.mass2motif.name]
            else:
                m2mpos = msmpos
                m2mindex[dm2m.mass2motif.name] = m2mpos
                msmpos += 1
            new_theta[m2mpos] = dm2m.probability
        theta_data.append(new_theta)

    pca = PCA(n_components = 2,whiten = True)
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
        new_value = (float(X[i,0]),float(X[i,1]),name,color)
        pca_data.append(new_value)

    # pca_data = []
    return HttpResponse(json.dumps(pca_data),content_type = 'application/json')

def validation(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    annotated_mass2motifs = []
    counts = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)
            dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = 0.05)
            tot = 0
            val = 0
            for d in dm2ms:
                tot += 1
                if d.validated:
                    val += 1
            counts.append((tot,val))
    annotated_mass2motifs = zip(annotated_mass2motifs,counts)
    context_dict = {'experiment':experiment}
    context_dict['annotated_mass2motifs'] = annotated_mass2motifs
    context_dict['counts'] = counts
    return render(request,'basicviz/validation.html',context_dict)


def toggle_dm2m(request,experiment_id,dm2m_id):
    dm2m = DocumentMass2Motif.objects.get(id = dm2m_id)
    jd = []
    if dm2m.validated:
        dm2m.validated = False
        jd.append('No')
    else:
        dm2m.validated = True
        jd.append('Yes')
    dm2m.save()

    return HttpResponse(json.dumps(jd),content_type = 'application/json')
    # return validation(request,experiment_id)

def dump_validations(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    annotated_mass2motifs = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="valid_dump_{}.csv"'.format(experiment_id)
    outstring = "msm_id,m2m_annotation,doc_id,doc_annotation,valid\n"
    writer = csv.writer(response)
    writer.writerow(['msm_id','m2m_annotation','doc_id','doc_annotation','valid'])
    for mass2motif in annotated_mass2motifs:
        dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = 0.05)
        for dm2m in dm2ms:
            # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
            writer.writerow([mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated])

    # return HttpResponse(outstring,content_type='text')
    return response
