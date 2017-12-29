import numpy as np
import pickle


def predict(classifier_object,test_data):
	bnb = pickle.loads(classifier_object.classifier)
	n_docs = len(test_data)
	motif_index = pickle.loads(classifier_object.feature_index)
	n_motifs = len(motif_index)

	test_array = np.zeros((n_docs,n_motifs),np.double)
	doc_index = {}
	doc_pos = 0
	for doc,mo in test_data.items():
		doc_index[doc] = doc_pos
		doc_pos += 1
		for motif,overlap_score in mo.items():
			motif_pos = motif_index.get(motif,None)
			if motif_pos: # why are some missing??
				test_array[doc_index[doc],motif_pos] = overlap_score

	probs = bnb.predict(test_array)
	print np.where(probs == 1)