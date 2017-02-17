import numpy as np
import bisect
import jsonpickle
from scipy.special import psi as psi

from decomposition.models import DocumentGlobalFeature,GlobalFeature,GlobalMotif

def decompose(documents,betaobject,normalise = 1000.0):
    # Load the beta objects
    print "Loading and unpickling beta"
    beta = jsonpickle.decode(betaobject.beta)
    alpha = jsonpickle.decode(betaobject.alpha_list)
    motif_id_list = jsonpickle.decode(betaobject.motif_id_list)
    feature_id_list = jsonpickle.decode(betaobject.feature_id_list)

    alpha_matrix = np.array(alpha)
    beta_matrix = np.array(beta)

    word_index = {}
    for i,word_id in enumerate(feature_id_list):
        feature = GlobalFeature.objects.get(id = word_id)
        word_index[feature] = i

    motif_index = {}
    motif_list = []
    for i,motif_id in enumerate(motif_id_list):
        motif = GlobalMotif.objects.get(id = motif_id)
        motif_index[motif] = i
        motif_list.append(motif)

    K = len(motif_list)
    print "Performing e-steps"
    for document in documents:
        print document.name
        docfeatures = DocumentGlobalFeature.objects.filter(document = document)

        doc_dict = {}
        maxi = 0.0
        for df in docfeatures:
            doc_dict[df.feature] = df.intensity
            if df.intensity > maxi:
                maxi = df.intensity

        # normalise
        if normalise:
            for word in doc_dict:
                doc_dict[word] = int(normalise*doc_dict[word]/maxi)

        phi_matrix = {}
        for word in doc_dict:
            phi_matrix[word] = None
        gamma = np.ones(K)
        for i in range(10): # do 100 iterations
            print "Iteration {}".format(i)
            temp_gamma = np.zeros(K) + alpha_matrix
            for word,intensity in doc_dict.items():
                # Find the word position in beta
                word_pos = word_index[word]
                if beta_matrix[:,word_pos].sum() > 0:
                    log_phi_matrix = np.log(beta_matrix[:,word_pos]) + psi(gamma).T
                    log_phi_matrix = np.exp(log_phi_matrix - log_phi_matrix.max())
                    phi_matrix[word] = log_phi_matrix/log_phi_matrix.sum()
                    temp_gamma += phi_matrix[word]*intensity

            gamma = temp_gamma.copy()

        # normalise the gamma
        theta = gamma/gamma.sum()
        theta = list(theta)

        theta_motif = zip(theta,motif_list)
        theta_motif = sorted(theta_motif,key = lambda x : x[0],reverse = True)
        theta,motif = zip(*theta_motif)
        for i in range(20):
            print theta[i],motif[i].originalmotif.name,motif[i].originalmotif.annotation

