# A collection of useful functions


# Creates a corpus in reverse: i.e. keys are features, values are a set of docs
def reverse_corpus(corpus):
	reverse_corpus = {}
	for doc,spectrum in corpus.items():
		for f,_ in spectrum.items():
			if not f in reverse_corpus:
				reverse_corpus[f] = set()
			reverse_corpus[f].add(doc)
	return reverse_corpus

# Counts how many docs each feature appears in
def count_docs(reverse_corpus,corpus,min_percent = 0.0,max_percent = 100.0):
	doc_counts = {}
	n_docs = len(corpus)
	for feature,docs in reverse_corpus.items():
		doc_counts[feature] = (len(docs),(100.0*len(docs))/n_docs)
	df = zip(doc_counts.keys(),doc_counts.values())
	to_remove = []
	if min_percent > 0.0:
		df2 = filter(lambda x: x[1][1] < min_percent,df)
		tr,_ = zip(*df2)
		to_remove += tr
	if max_percent < 100.0:
		df2 = filter(lambda x: x[1][1] > max_percent,df)
		tr,_ = zip(*df2)
		to_remove += tr
	to_remove = set(to_remove)

	return doc_counts,to_remove

# remove a particular set of features from the corpus
def remove_features(corpus,to_remove):
	for doc,spectrum in corpus.items():
		features = set(spectrum.keys())
		overlap = features.intersection(to_remove)
		for f in overlap:
			del spectrum[f]
	return corpus


def bin_mass(mass,bin_width = 0.005):
	import numpy as np
	# return the name of the bin center for the mass given the specified bin width
	bin_no = np.floor(mass / bin_width)
	bin_lower = bin_no*bin_width
	bin_upper = bin_lower + bin_width
	bin_middle = (bin_lower + bin_upper)/2.0
	return "{:.4f}".format(bin_middle)