# Some methods required by annotation
import numpy as np
from scipy.special import psi as psi
from basicviz.models import Experiment,Feature,Mass2Motif,Alpha,Mass2MotifInstance

def annotate(spectrum,basicviz_experiment_id):
    parentmass = spectrum[0]
    document = create_document_dictionary(spectrum,basicviz_experiment_id)
    document = normalise_document(document,max_intensity = 1000.0)
    motif_theta_overlap,plotdata = do_e_steps(parentmass,document,basicviz_experiment_id)

    doc_list = zip(document.keys(),document.values())
    doc_list = sorted(doc_list,key = lambda x: x[1],reverse = True)

    return doc_list,motif_theta_overlap,plotdata



def create_document_dictionary(spectrum,basicviz_experiment_id):
    experiment = Experiment.objects.get(id = basicviz_experiment_id)
    peaks = spectrum[1]
    parentmass = spectrum[0]
    features = Feature.objects.filter(experiment = experiment).order_by('max_mz')


    fragments = [f for f in features if f.name.startswith('fragment')]
    losses = [f for f in features if f.name.startswith('loss')]

    print "Found {} features ({} fragments, {} losses)".format(len(features),len(fragments),len(losses))

    document = {}

    for p in peaks:
        print "Searching for {}".format(p[0])
        pos = 0
        fragment_mass = p[0]
        while True:
            if fragment_mass >= fragments[pos].min_mz and fragment_mass <= fragments[pos].max_mz:
                document[fragments[pos]] = p[1]
                print "\tFound fragment"
                break
            pos += 1
            if pos >= len(fragments):
                break
            if fragment_mass < fragments[pos].min_mz:
                break

        pos = 0
        loss_mass = parentmass - p[0]
        while True:
            if loss_mass >= losses[pos].min_mz and loss_mass <= losses[pos].max_mz:
                document[losses[pos]] = p[1]
                print "\tFound loss"
                break
            pos += 1
            if pos >= len(losses):
                break
            if loss_mass < losses[pos].min_mz:
                break
    print document
    return document


def normalise_document(document,max_intensity = 1000.0):
    maxi = 0.0
    for feature in document:
        if document[feature] >= maxi:
            maxi = document[feature]
    for feature in document:
        document[feature] = int(max_intensity*document[feature]/maxi)
    return document


def do_e_steps(parentmass,document,basicviz_experiment_id,nsteps = 500):
    experiment = Experiment.objects.get(id = basicviz_experiment_id)
    motifs = Mass2Motif.objects.filter(experiment = experiment).order_by('name')
    K = len(motifs) # The number of motifs
    features = document.keys()
    sub_beta = np.zeros((len(motifs),len(features)))
    alpha = []
    for motif in motifs:
        alpha.append(Alpha.objects.get(mass2motif = motif).value)
    alpha =np.array(alpha)
    # alpha = np.array([1.0 for m in motifs]) # This seems to make things work better? But alpha should probably be the learnt value...

    print "Got {} motifs".format(len(motifs))
    print "Got {} alphas".format(len(alpha))
    print "Got {} features".format(len(features))

    word_index = {}
    for i,motif in enumerate(motifs):
        if motif.name == 'motif_10':
            pos10 = i
        if motif.name == 'motif_88':
            pos88 = i
        for j,feature in enumerate(features):
            word_index[feature] = j
            try:
                mi = Mass2MotifInstance.objects.get(mass2motif = motif,feature = feature)
                print motif,feature
                sub_beta[i,j] = mi.probability
            except:
                sub_beta[i,j] = 0.0 # Small value to avoid numerical errors
    print sub_beta.sum(axis=0)


    gamma = np.ones(K)
    phi = {}
    for word in document:
        phi[word] = np.zeros(K)


    print sub_beta[pos10,:]
    print sub_beta[pos88,:]
    print alpha[pos10],alpha[pos88]

    for step in range(nsteps):
        print step
        temp_gamma = np.zeros(K) + alpha
        for word in document:
            word_pos = word_index[word]
            phi[word] = sub_beta[:,word_pos]*np.exp(psi(gamma)-psi(gamma.sum())).T
            s = phi[word].sum()
            if s > 0.0:
                phi[word] /= s
            temp_gamma += phi[word]*document[word]
        gamma = temp_gamma.copy()


    theta = gamma/gamma.sum()
    theta_float = [float(t) for t in theta]
    overlap_scores = compute_overlap_score(phi,motifs,sub_beta,word_index)
    mto = zip(motifs,theta_float,overlap_scores)
    plotdata = make_plot_object(document,parentmass,phi,theta,motifs)
    return mto,plotdata

def compute_overlap_score(phi,motifs,sub_beta,word_index):
    K = len(motifs)
    scores = np.zeros(K)
    for word in phi:
        word_pos = word_index[word]
        scores += sub_beta[:,word_pos]*phi[word]
    return [float(s) for s in scores]

def make_plot_object(document,parentmass,phi,thetas,motifs):
    colors = ['red','blue','green','black'] # top 4 motifs
    thetas_copy = thetas.copy()
    use_motifs = []
    for i in range(4):
        max_pos = thetas_copy.argmax()
        use_motifs.append((max_pos,motifs[max_pos]))
        thetas_copy[max_pos] = 0.0


    plotdata = []
    parent_data = (parentmass, 100.0, "test document", "", "null")
    plotdata.append(parent_data)


    maxi = 0.0
    for feature in document:
        if document[feature] > maxi:
            maxi = document[feature]


    child_data = []
    for feature in phi:
        total_intensity = 100.0*document[feature]/maxi
        if feature.name.startswith('fragment'):
            mz = float(feature.name.split('_')[1])
            current_pos = 0.0
            for j,(i,motif) in enumerate(use_motifs):
                prob = phi[feature][i]
                end_pos = current_pos + total_intensity * prob
                child_data.append((mz,mz,current_pos,end_pos,1,colors[j],feature.name))
                current_pos = end_pos
            child_data.append((mz,mz,current_pos,total_intensity,1,'gray',"other topics"))
        if feature.name.startswith('loss'):
            mz = float(feature.name.split('_')[1])
            current_pos = parentmass - mz
            total_width = mz
            for j,(i,motif) in enumerate(use_motifs):
                prob = phi[feature][i]
                end_pos = current_pos + total_width * prob
                child_data.append((current_pos,end_pos,total_intensity,total_intensity,0,colors[j],feature.name))
                current_pos = end_pos
            child_data.append((current_pos,parentmass,total_intensity,total_intensity,0,'gray','other topics'))

    plotdata.append(child_data)

    key = []
    for j,(i,motif) in enumerate(use_motifs):
        key.append((motif.name, colors[j]))

    return [key,[plotdata]]

