import numpy as np
import pickle


def predict(classifier_object,test_data):
	# Creates the test data array and gets the predictions from the model
	if isinstance(test_data,list):
		# turn into a dict
		td = {}
		for doc_name,doc_motifs in test_data:
			td[doc_name] = {}
			for motif,o in doc_motifs:
				td[doc_name][motif] = o
		test_data = td
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


	probs = bnb.predict_proba(test_array)
	output = {}

	for doc,doc_pos in doc_index.items():
		output[doc] = {}
		for i,cl in enumerate(bnb.classes_):
			output[doc][cl] = probs[doc_pos,i]

	return output