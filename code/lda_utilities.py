import numpy as np
import pickle
import jsonpickle

def match_topics_across_dictionaries(lda1 = None,lda2 = None,file1 = None,file2 = None,
	same_corpus = True,copy_annotations = False,copy_threshold = 0.5,summary_file = None,
	new_file2 = None,mass_tol = 5.0):
	# finds the closest topic matches from lda2 to lda1
	if lda1 == None:
		if file1 == None:
			print "Must specify either an lda dictionary object or a dictionary file for lda1"
			return
		else:
			with open(file1,'r') as f:
				lda1 = pickle.load(f)
				print "Loaded lda1 from {}".format(file1)
	if lda2 == None:
		if file2 == None:
			print "Must specify either an lda dictionary object or a dictionary file for lda1"
			return
		else:
			with open(file2,'r') as f:
				lda2 = pickle.load(f)
				print "Loaded lda2 from {}".format(file2)


	word_index = lda1['word_index']
	n_words = len(word_index)
	n_topics1 = lda1['K']
	n_topics2 = lda2['K']

	# Put lda1's topics into a nice matrix
	beta = np.zeros((n_topics1,n_words),np.float)
	topic_pos = 0
	topic_index1 = {}
	for topic in lda1['beta']:
		topic_index1[topic] = topic_pos
		for word in lda1['beta'][topic]:
			word_pos = word_index[word]
			beta[topic_pos,word_pos] = lda1['beta'][topic][word]
		topic_pos += 1

	# Make the reverse index
	ti = [(topic,topic_index1[topic]) for topic in topic_index1]
	ti = sorted(ti,key = lambda x: x[1])
	reverse1,_ = zip(*ti)

	if not same_corpus:
		fragment_masses = np.array([float(f.split('_')[1]) for f in word_index if f.startswith('fragment')])
		fragment_names = [f for f in word_index if f.startswith('fragment')]
		loss_masses = np.array([float(f.split('_')[1]) for f in word_index if f.startswith('loss')])
		loss_names = [f for f in word_index if f.startswith('loss')]

	beta /= beta.sum(axis=1)[:,None]
	best_match = {}
	temp_topics2 = {}
	for topic2 in lda2['beta']:
		temp_topics2[topic2] = {}
		temp_beta = np.zeros((1,n_words))
		if same_corpus:
			total_probability = 0.0
			for word in lda2['beta'][topic2]:
				word_pos = word_index[word]
				temp_beta[0,word_pos] = lda2['beta'][topic2][word]
				temp_topics2[topic2][word] = lda2['beta'][topic2][word]
				total_probability += temp_topics2[topic2][word]
			for word in temp_topics2[topic2]:
				temp_topics2[topic2][word] /= total_probability
			temp_beta /= temp_beta.sum(axis=1)[:,None]
		else:
			# we need to match across corpus
			total_probability = 0.0
			for word in lda2['beta'][topic2]:
				# try and match to a word in word_index
				split_word = word.split('_')
				word_mass = float(split_word[1])
				if split_word[0].startswith('fragment'):
					ppm_errors = 1e6*np.abs((fragment_masses - word_mass)/fragment_masses)
					smallest_pos = ppm_errors.argmin()
					if ppm_errors[smallest_pos] < mass_tol:
						word1 = fragment_names[smallest_pos]
						temp_topics2[topic2][word1] = lda2['beta'][topic2][word]
						temp_beta[0,word_index[word1]] = lda2['beta'][topic2][word]
				if split_word[0].startswith('loss'):
					ppm_errors = 1e6*np.abs((loss_masses - word_mass)/loss_masses)
					smallest_pos = ppm_errors.argmin()
					if ppm_errors[smallest_pos] < 2*mass_tol:
						word1 = loss_names[smallest_pos]
						temp_topics2[topic2][word1] = lda2['beta'][topic2][word]
						temp_beta[0,word_index[word1]] = lda2['beta'][topic2][word]
				total_probability += lda2['beta'][topic2][word]
			for word in temp_topics2[topic2]:
				temp_topics2[topic2][word] /= total_probability
			temp_beta /= total_probability


		
		
		match_scores = np.dot(beta,temp_beta.T)
		best_score = match_scores.max()
		best_pos = match_scores.argmax()

		topic1 = reverse1[best_pos]
		w1 = lda1['beta'][topic1].keys()
		if same_corpus:
			w2 = lda2['beta'][topic2].keys()
		else:
			w2 = temp_topics2[topic2].keys()
		union = set(w1) | set(w2)
		intersect = set(w1) & set(w2)
		p1 = 0.0
		p2 = 0.0
		for word in intersect:
			word_pos = word_index[word]
			p1 += beta[topic_index1[topic1],word_pos]
			p2 += temp_topics2[topic2][word]

		annotation = ""
		if 'topic_metadata' in lda1:
			if topic1 in lda1['topic_metadata']:
				if type(lda1['topic_metadata'][topic1]) == str:
					annotation = lda1['topic_metadata'][topic1]
				else:
					annotation = lda1['topic_metadata'][topic1].get('annotation',"")
		best_match[topic2] = (topic1,best_score,len(union),len(intersect),p2,p1,annotation)

	if summary_file:
		with open(summary_file,'w') as f:
			f.write('lda2_topic,lda1_topic,match_score,unique_words,shared_words,shared_p_lda2,shared_p_lda1,lda1_annotation\n')
			for topic2 in best_match:
				topic1 = best_match[topic2][0]
				line = "{},{},{}".format(topic2,topic1,best_match[topic2][1])
				line += ",{},{}".format(best_match[topic2][2],best_match[topic2][3])
				line += ",{},{}".format(best_match[topic2][4],best_match[topic2][5])		
				line += ",{}".format(best_match[topic2][6])
				f.write(line+'\n')


	if copy_annotations and 'topic_metadata' in lda1:
		print "Copying annotations"
		if not 'topic_metadata' in lda2:
			lda2['topic_metadata'] = {}
		for topic2 in best_match:
			lda2['topic_metadata'][topic2] = {'name':topic2}
			topic1 = best_match[topic2][0]
			p2 = best_match[topic2][4]
			p1 = best_match[topic2][5]
			if p1 >= copy_threshold and p2 >= copy_threshold:
				annotation = best_match[topic2][6]
				if len(annotation) > 0:
					lda2['topic_metadata'][topic2]['annotation'] = annotation
		if new_file2 == None:
			with open(file2,'w') as f:
				pickle.dump(lda2,f)
			print "Dictionary with copied annotations saved to {}".format(file2)
		else:
			with open(new_file2,'w') as f:
				pickle.dump(lda2,f)
			print "Dictionary with copied annotations saved to {}".format(new_file2)


	return best_match,lda2


def find_standards_in_dict(standards_file,lda_dict=None,lda_dict_file=None,mode='pos',mass_tol = 3,rt_tol = 12,new_lda_file = None):
	if lda_dict == None:
		if lda_dict_file == None:
			print "Must provide either an lda dictionary or an lda dictionary file"
			return
		else:
			with open(lda_dict_file,'r') as f:
				lda_dict = pickle.load(f)
			print "Loaded lda dictionary from {}".format(lda_dict_file)

	# Load the standards
	standard_molecules = []
	found_heads = False
	with open(standards_file,'r') as f:
		for line in f:
			if found_heads == False and line.startswith('Peak Num'):
				found_heads = True
				continue
			elif found_heads == False:
				continue
			else:
				split_line = line.rstrip().split(',')
				if (mode == 'pos' and split_line[4] == '+') or (mode == 'neg' and split_line[3] == '-'):
					# It's a keeper
					name = split_line[2]
					mz = split_line[6]
					if mz == 'N':
						continue
					mz = float(mz)
					rt = split_line[9]
					if rt == '-':
						continue
					rt = float(rt)*60.0 # converted to seconds
					formula = split_line[3]
					standard_molecules.append((name,mz,rt,formula))
					# mol = ()
	print "Loaded {} molecules".format(len(standard_molecules))
	
	doc_masses = np.array([float(d.split('_')[0]) for d in lda_dict['corpus']])
	doc_names = [d for d in lda_dict['corpus']]
	doc_rt = np.array([float(d.split('_')[1]) for d in lda_dict['corpus']])
	hits = {}
	for mol in standard_molecules:
		mass_delta = mol[1]*mass_tol*1e-6
		mass_hit = (doc_masses < mol[1] + mass_delta) & (doc_masses > mol[1] - mass_delta)
		rt_hit = (doc_rt < mol[2] + rt_tol) & (doc_rt > mol[2] - rt_tol)
		match = np.where(mass_hit & rt_hit)[0]
		
		if len(match) > 0:
			if len(match) == 1:
				hits[mol] = doc_names[match[0]]
			else:
				# Multiple hits
				min_dist = 1e6
				best_match = match[0]
				for individual_match in match:
					match_mass = doc_masses[individual_match]
					match_rt = doc_rt[individual_match]
					dist = np.sqrt((match_rt - mol[2])**2 + (match_mass - mol[1])**2)
					if dist < min_dist:
						best_match = individual_match
				hits[mol] = doc_names[best_match]
	

	print "Found hits for {} standard molecules".format(len(hits))
	# Add the hits to the lda_dict as document metadata
	for mol in hits:
		doc_name = hits[mol]
		lda_dict['doc_metadata'][doc_name]['standard_mol'] = mol[0]
		lda_dict['doc_metadata'][doc_name]['annotation'] = mol[0]

	if new_lda_file:
		with open(new_lda_file,'w') as f:
			pickle.dump(lda_dict,f)
		print "Wrote annotated dictionary to {}".format(new_lda_file)

	return lda_dict


def alpha_report(vlda):
	ta = []
	for topic,ti in vlda.topic_index.items():
		ta.append((topic,vlda.alpha[ti]))
	ta = sorted(ta,key = lambda x: x[1],reverse = True)
	for t,a in ta:
		print t,vlda.topic_metadata[t].get('SHORT_ANNOTATION',None),a