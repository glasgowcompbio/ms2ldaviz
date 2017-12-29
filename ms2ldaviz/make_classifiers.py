import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()
from django.db import transaction
import sys
import pickle

from scipy.sparse import coo_matrix

from sklearn.naive_bayes import BernoulliNB


from annotation.models import SubstituentTerm
from basicviz.models import Experiment,Document,DocumentMass2Motif
from term_classifier.models import SubClassifier
from term_classifier.classification_functions import predict

def make_classifiers():
    e = Experiment.objects.get(name = 'massbank_binned_005')
    SubClassifier.objects.filter(experiment = e).delete()
    train_doc_file  = '/Users/simon/Dropbox/BioResearch/Meta_clustering/MS2LDA/classyfire/massbank_005_server.dict'
    with open(train_doc_file,'r') as f:
        train_doc_data = pickle.load(f)
    with open('/Users/simon/Dropbox/BioResearch/Meta_clustering/MS2LDA/classyfire/massbank_classyfire_corpora.dict','r') as f:
        train_cc = pickle.load(f)

    term_type = 'substituents'

    train_list = train_cc[term_type + '_list']
    train_corpus = train_cc[term_type + '_corpus']

    import numpy as np


    motif_index = {}
    motif_pos = 0
    doc_index = {}
    doc_pos = 0
    sparse_data = []
    for doc_name,doc_frags,doc_motifs in train_doc_data:
        doc_index[doc_name] = doc_pos
        for motif,p,o in doc_motifs:
            if not motif in motif_index:
                motif_index[motif] = motif_pos
                motif_pos += 1
            sparse_data.append((doc_index[doc_name],motif_index[motif],p,o))
        
        doc_pos += 1
    d,m,p,o = zip(*sparse_data)
    train_p = np.array(coo_matrix((p,(d,m)),shape=[len(doc_index),len(motif_index)]).todense())
    train_o = np.array(coo_matrix((o,(d,m)),shape=[len(doc_index),len(motif_index)]).todense())

    sparse_term_data = []
    term_index = {}
    term_pos = 0
    for doc in train_corpus:
        for pos,termval in enumerate(train_corpus[doc]):
            if termval == 1:
                term = train_list[pos]
                if not term in term_index:
                    term_index[term] = term_pos
                    term_pos += 1
                sparse_term_data.append((doc_index[doc],term_index[term],1))

    d,t,v = zip(*sparse_term_data)
    train_terms = np.array(coo_matrix((v,(d,t)),shape=[len(doc_index),len(term_index)]).todense())


    for term in term_index:
        print "Making classifier for {}".format(term)
        term_pos = term_index[term]
        y = train_terms[:,term_pos]

        n_train = len(doc_index)
        n_term = train_terms[:,term_pos].sum()
        proportion = (1.0*n_term)/(1.0*n_train)
        if proportion >= 0.001 and proportion <= 0.8:

            bnb = BernoulliNB(alpha = .1,binarize = 0.05,fit_prior=True)
            bnb.fit(train_o,y)

            subterm = SubstituentTerm.objects.get(name = term)

            s,_ = SubClassifier.objects.get_or_create(term = subterm)
            s.classifier = pickle.dumps(bnb)
            s.feature_index = pickle.dumps(motif_index)
            s.classifier_type = 'sklearn.naive_bayes.BernoulliNB'
            s.experiment = e
            s.save()
    


def test_classifier(classifier_model):
    e = Experiment.objects.get(name = 'massbank_binned_005')
    docs = Document.objects.filter(experiment = e)
    test_data = {}
    for doc in docs:
        m2ms = DocumentMass2Motif.objects.filter(document = doc)
        test_data[doc.name] = {}
        for m in m2ms:
            test_data[doc.name][m.mass2motif.name] = m.overlap_score

    output = predict(classifier_model,test_data)
    print output




if __name__ == '__main__':
    # load the data
    # make_classifiers()
    test_classifier(SubClassifier.objects.all()[100])
