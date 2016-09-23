import sys
import time
import pymzml
import numpy as np
import pylab as plt

sys.path.append('../code')
from lda import VariationalLDA, MultiFileVariationalLDA

prefix = '/Users/joewandy/Dropbox/Meta_clustering/MS2LDA/MS2LDA Manuscript Sections/Matrices/Beer3pos_MS1filter_Method3'
total_lda = VariationalLDA(K = 50,eta=0.1,alpha=1)
total_lda.load_features_from_csv(prefix,scale_factor=100.0)

corpus_list = []
corpus_1 = {}
corpus_2 = {}
for doc in total_lda.corpus:
    if np.random.choice([0,1]) == 0:
        corpus_1[doc] = total_lda.corpus[doc]
    else:
        corpus_2[doc] = total_lda.corpus[doc]
print len(corpus_1),len(corpus_2)
corpus_list.append(corpus_1)
corpus_list.append(corpus_2)

corpus_dictionary = {}
corpus_dictionary[0] = corpus_1
corpus_dictionary[1] = corpus_2
word_index = total_lda.word_index

mf_lda = MultiFileVariationalLDA(corpus_dictionary, word_index, K = 300, alpha=1, eta=0.1)

start_time = time.clock()
mf_lda.run_vb(parallel=False, n_its=10)
end_time = time.clock()
print 'total time', end_time - start_time

start_time = time.clock()
mf_lda.run_vb(parallel=True, n_its=10)  
end_time = time.clock()
print 'total time', end_time - start_time
