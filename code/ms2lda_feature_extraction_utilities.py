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
