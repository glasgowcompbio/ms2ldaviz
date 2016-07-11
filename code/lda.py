# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import psi as psi
from scipy.special import polygamma as pg

import plotly as plotly
from plotly.graph_objs import *

# This is a Gibbs sampler LDA object. Don't use it. I'll probably delete it when I have time
class LDA(object):
	def __init__(self,corpus,K=20,alpha=1,beta=1):
		self.corpus = corpus
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.collect_words()
		self.initialise()

	def collect_words(self):
		self.words = []
		self.nwords = 0
		self.ndocs = len(self.corpus)
		docpos = 0
		self.doc_index = {}
		self.word_index = {}
		for doc in self.corpus:
			self.doc_index[doc] = docpos
			docpos += 1
			for word in self.corpus[doc]:
				if not word in self.word_index:
					self.word_index[word] = self.nwords
					self.nwords += 1

		


	def initialise(self):
		self.Z = {}
		self.doc_topic_counts = np.zeros((self.K,self.ndocs),np.int) + self.alpha
		self.topic_word_counts = np.zeros((self.K,self.nwords),np.int) + self.beta
		self.topic_totals = np.zeros((self.K),np.int) + self.beta
		self.total_words = 0
		self.word_counts = {}
		for word in self.word_index:
			self.word_counts[word] = 0

		for doc in self.corpus:
			self.Z[doc] = {}
			di = self.doc_index[doc]
			for word in self.corpus[doc]:
				wi = self.word_index[word]
				count = self.corpus[doc][word]
				self.total_words += count
				self.word_counts[word] += count
				self.Z[doc][word] = []
				for c in range(count):
					topic = np.random.randint(self.K)
					self.topic_totals[topic] += 1
					self.Z[doc][word].append(topic)
					self.doc_topic_counts[topic,di] += 1
					self.topic_word_counts[topic,wi] += 1

		# Output things
		self.post_sample_count = 0.0
		self.post_mean_theta = np.zeros((self.K,self.ndocs),np.float)
		self.post_mean_topics = np.zeros((self.K,self.nwords),np.float)


	def gibbs_iteration(self,n_samples = 1,verbose = True,burn = True):
		# Does one gibbs step
		for sample in range(n_samples):
			if verbose:
				print "Sample {} of {} (Burn is {})".format(sample,n_samples,burn)
			for doc in self.corpus:
				di = self.doc_index[doc]
				for word in self.corpus[doc]:
					wi = self.word_index[word]
					for i,instance in enumerate(self.Z[doc][word]):
						current_topic = instance
						self.doc_topic_counts[current_topic,di] -= 1
						self.topic_word_counts[current_topic,wi] -= 1
						self.topic_totals[current_topic] -= 1

						# Re-sample
						p_topic = 1.0*self.topic_word_counts[:,wi] / self.topic_totals
						p_topic *= self.doc_topic_counts[:,di]
						p_topic = 1.0*p_topic / p_topic.sum()
						new_topic = np.random.choice(self.K,p=p_topic)

						self.Z[doc][word][i] = new_topic

						self.doc_topic_counts[new_topic,di] += 1
						self.topic_word_counts[new_topic,wi] += 1
						self.topic_totals[new_topic] += 1

		if not burn:
			self.post_sample_count += 1.0
			for doc in self.corpus:
				di = self.doc_index[doc]
				tcounts = self.doc_topic_counts[:,di]
				self.post_mean_theta[:,di] += np.random.dirichlet(tcounts)
			for topic in range(self.K):
				wcounts = self.topic_word_counts[topic,:]
				self.post_mean_topics[topic,:] += np.random.dirichlet(wcounts)
			
	def get_post_mean_theta(self):
		return self.post_mean_theta / self.post_sample_count
	def get_post_mean_topics(self):
		return self.post_mean_topics / self.post_sample_count

	def get_mass_plot(self,topic_id):
		pmeantopics = self.get_post_mean_topics()
		m = []
		probs = []
		for word in self.word_index:
			m.append(float(word))
			probs.append(pmeantopics[topic_id,self.word_index[word]])

		m_probs = zip(m,probs)
		m_probs = sorted(m_probs,key=lambda x: x[0])
		m,probs = zip(*m_probs)
		return np.array(m),np.array(probs)


	def plot_topic(self,topic_id,nrows = 10,ncols = 10):

		image_array = np.zeros((nrows,ncols),np.float)
		for doc in self.corpus:
			di = self.doc_index[doc]
			if self.post_sample_count == 0:
				tprobs = self.doc_topic_counts[:,di]
				tprobs = tcounts / 1.0*tcounts.sum()
			else:
				tprobs = self.get_post_mean_theta()
			(r,c) = doc
			image_array[r,c] = tprobs[topic_id,di]

		return image_array

	def get_topic_as_dict(self,topic_id,thresh = 0.001):
		pmt = self.get_post_mean_topics()
		top = {}
		for word in self.word_index:
			pos = self.word_index[word]
			if pmt[topic_id,pos] >= thresh:
				top[word] = pmt[topic_id,pos]
		return top

	def get_topic_as_doc_dict(self,topic_id,thresh = 0.001):
		pmth = self.get_post_mean_theta()
		top = {}
		for doc in self.doc_index:
			pos = self.doc_index[doc]
			if pmth[topic_id,pos] >= thresh:
				top[doc] = pmth[topic_id,pos]
		return top
		
	def get_topic_as_tuples(self,topic_id,thresh = 0.001):
		pmth = self.get_post_mean_topics()
		top = []
		for word in self.word_index:
			pos = self.word_index[word]
			if pmth[topic_id,pos] >= thresh:
				top.append((word,pmth[topic_id,pos]))

		return sorted(top,key = lambda x: x[1], reverse=True)


# This is code for parsing an mzml file, only used by the DESI imagining work.
# Will probably remove and put in a different repo at some point
class LDA_Feature_Extractor(object):
	def __init__(self,filename,use_scans = 'even',tol = 50, min_intense = 500, min_occurance = 5, max_occurance = 200,min_mass = 50.0,max_mass = 300.0,min_doc_word_instances = 5,max_doc_word_instances = 200):
		self.tol = tol
		self.min_intense = min_intense
		self.min_occurance = min_occurance
		self.max_occurance = max_occurance
		self.filename = filename
		self.min_mass = min_mass
		self.max_mass = max_mass
		self.use_scans = use_scans
		self.min_doc_word_instances = min_doc_word_instances
		self.max_doc_word_instances = max_doc_word_instances


	def make_corpus(self):
		import pymzml
		total_peaks = 0
		self.word_masses = []
		self.word_names = []
		self.instances = []
		self.total_m = []
		run = pymzml.run.Reader(self.filename,MS1_Precision = 5e-6)
		self.corpus = {}
		spec_pos = 0
		for spectrum in run:
			if self.use_scans == 'even' and spec_pos % 2 == 1:
				spec_pos += 1
				continue
			if self.use_scans == 'odd' and spec_pos % 2 == 0:
				spec_pos += 1
				continue
			new_doc = {}
			max_i = 3000.0 
			min_i = 1e10
			for m,i in spectrum.peaks:
				if i >= self.min_intense and m >= self.min_mass and m <= self.max_mass:
					word = None
					if len(self.word_masses) == 0:
						self.word_masses.append(m)
						self.word_names.append(str(m))
						self.instances.append(1)
						self.total_m.append(m)
						word = str(m)
					else:
						idx = np.abs(m - np.array(self.word_masses)).argmin()
						if not self.hit(m,self.word_masses[idx],self.tol):
							self.word_masses.append(m)
							self.word_names.append(str(m))
							self.instances.append(1)
							self.total_m.append(m)
							word = str(m)
						else:
							self.total_m[idx] += m
							self.instances[idx] += 1
							# self.word_masses[idx] = self.total_m[idx]/self.instances[idx]
							# self.word_names[idx] = str(self.word_masses[idx])
							word = self.word_names[idx]
					if word in new_doc:
						new_doc[word] += i
					else:
						new_doc[word] = i
					if i < min_i:
						min_i = i

			# to_remove = []
			# for word in new_doc:
			# 	if new_doc[word] > max_i:
			# 		new_doc[word] = max_i
			# 	new_doc[word] -= min_i
			# 	new_doc[word] /= (max_i - min_i)
			# 	new_doc[word] *= 100.0
			# 	new_doc[word] = int(new_doc[word])
			# 	if new_doc[word] == 0:
			# 		to_remove.append(word)

			# for word in to_remove:
			# 	del new_doc[word]

			self.corpus[str(spec_pos)] = new_doc
			spec_pos += 1
			if spec_pos % 100 == 0:
				print "Spectrum {} ({} words)".format(spec_pos,len(self.word_names))


		print "Found {} documents".format(len(self.corpus))

		word_counts = {}
		for doc in self.corpus:
			for word in self.corpus[doc]:
				if word in word_counts:
					word_counts[word] += 1
				else:
					word_counts[word] = 1



		to_remove = []
		for word in word_counts:
			if word_counts[word] < self.min_doc_word_instances:
				to_remove.append(word)
			if word_counts[word] > self.max_doc_word_instances:
				to_remove.append(word)


		print "Removing {} words".format(len(to_remove))

		for doc in self.corpus:
			for word in to_remove:
				if word in self.corpus[doc]:
					del self.corpus[doc][word]


	def make_nominal_corpus(self):
		import pymzml
		self.word_names = []
		self.word_masses = []
		self.word_names = []
		run = pymzml.run.Reader(self.filename,MS1_Precision = 5e-6)
		self.corpus = {}
		spec_pos = 0
		for spectrum in run:
			doc = str(spec_pos)
			self.corpus[doc] = {}
			for m,i in spectrum.peaks:
				if m >= self.min_mass and m <= self.max_mass and i >= self.min_intense:
					word = str(np.floor(m))
					if not word in self.word_names:
						self.word_names.append(word)
						self.word_masses.append(float(word))

					if word in self.corpus[doc]:
						self.corpus[doc][word] += i
					else:
						self.corpus[doc][word] = i
			spec_pos += 1



	def hit(self,m1,m2,tol):
	    if 1e6*abs(m1-m2)/m1 < tol:
	        return True
	    else:
	        return False


# This is the LDA implementation to use
# Corpus can be passed in, or loaded from .csv files in joes style
# K = number of topics
# eta = hyperparameter for topics (i.e. pseudo word counts)
# alpha = initial Dirichlet hyperparameter
# update_alpha = boolean to determine whether or not alpha is updated at each iteration
# word_index is a dictionary storing the position of each feature in numpy arrays
 # word_index is only used in multi-file as it's important that features are always in the same order.
 # In single file it is created internally
class VariationalLDA(object):
	def __init__(self,corpus=None,K = 20,eta=0.1,alpha=1,update_alpha=True,word_index=None):
		self.corpus = corpus
		self.word_index = word_index
		#  If the corpus exists, make the word index and the (unused?) word doc matrix
		if not self.corpus == None:
			self.n_docs = len(self.corpus)
			if self.word_index == None:
				self.word_index = self.find_unique_words()
			print "Object created with {} documents".format(self.n_docs)
			self.n_words = len(self.word_index)

			self.make_doc_index()
		
		self.K = K
		self.alpha = alpha
		#  If alpha is a single value, make it into a vector
		if type(self.alpha) == int or type(self.alpha) == float:
			self.alpha = self.alpha*np.ones(self.K)
		self.eta = eta # Smoothing parameter for beta
		self.update_alpha = update_alpha

    # Load the features from a Joe .csv file. Pass the file name up until the _ms1.csv or _ms2.csv
    # these are added here
    # The scale factor is what we multiply intensities by
	def load_features_from_csv(self,prefix,scale_factor=100.0):
		# Load the MS1 peaks (MS1 object defined below)
		self.ms1peaks = []
		ms1file = prefix + '_ms1.csv'
		with open(ms1file,'r') as f:
		    heads = f.readline()
		    for line in f:
		        split_line = line.split(',')
		        ms1_id = split_line[1]
		        mz = float(split_line[5])
		        rt = float(split_line[4])
		        name = split_line[5] + '_' + split_line[4]
		        intensity = float(split_line[6])
		        new_ms1 = MS1(ms1_id,mz,rt,intensity,name)
		        self.ms1peaks.append(new_ms1)
		print "Loaded {} MS1 peaks".format(len(self.ms1peaks))
		parent_id_list = [i.ms1_id for i in self.ms1peaks]


		# Load the ms2 objects
		frag_file = prefix + '_ms2.csv'
		features = []
		self.corpus = {}
		with open(frag_file,'r') as f:
		    heads = f.readline().split(',')
		    for line in f:
		        split_line = line.rstrip().split(',')
		        frag_name = split_line[10]
		        if not frag_name == 'NA':
		            frag_name = frag_name[1:-1]
		        frag_id = 'fragment_' + frag_name
		        
		        loss_name = split_line[11]
		        if not loss_name == 'NA':
		            loss_name = loss_name[1:-1]
		            loss_id = 'loss_' + loss_name
		        
		        if not frag_id == "fragment_NA":
		            if not frag_id in features:
		                features.append(frag_id)
		            frag_idx = features.index(frag_id)

		        if not loss_id == "loss_NA":
		            if not loss_id in features:
		                features.append(loss_id)
		            loss_idx = features.index(loss_id)
		        
		        intensity = float(split_line[6])
		        
		        parent_id = split_line[2]
		        # Find the parent
		        parent = self.ms1peaks[parent_id_list.index(parent_id)]

		        # If we've not seen this parent before, create it as an empty dict
		        if not parent in self.corpus:
		            self.corpus[parent] = {}

		        # Store the ms2 features in the parent dictionary
		        if not frag_id == "fragment_NA":
		            self.corpus[parent][frag_id] = intensity * scale_factor
		        if not loss_id == "loss_NA":
		            self.corpus[parent][loss_id] = intensity * scale_factor

		self.n_docs = len(self.corpus)
		if self.word_index == None:
			self.word_index = self.find_unique_words()
		print "Object created with {} documents".format(self.n_docs)
		self.n_words = len(self.word_index)

		# I don't think this does anything - I will check
		self.make_doc_index()

	# Run the VB inference. Verbose = True means it gives output each iteration
	# initialise = True initialises (i.e. restarts the algorithm)
	# This means we can run the algorithm from where it got to.
	# First time its run, initialise has to be True
	def run_vb(self,n_its = 1,verbose=True,initialise=True):
		if initialise:
			print "Initialising"
			self.init_vb()
		print "Starting iterations"
		for it in range(n_its):
			diff = self.vb_step()
			if verbose:
				print "Iteration {} (change = {})".format(it,diff)


	# D a VB step
	def vb_step(self):
		# Run an e-step
		temp_beta = self.e_step()
		temp_beta += self.eta
		# Do the normalisation in the m step
		temp_beta /= temp_beta.sum(axis=1)[:,None]
		# Compute how much the word probabilities have changed
		total_difference = (np.abs(temp_beta - self.beta_matrix)).sum()
		self.beta_matrix = temp_beta
		# If we're updating alpha, run the alpha update
		if self.update_alpha:
			self.alpha = self.alpha_nr()
		return total_difference
		# self.m_step()



	# Newton-Raphson procedure for updating alpha
	def alpha_nr(self,maxit=20,init_alpha=[]):
	    M,K = self.gamma_matrix.shape
	    if not len(init_alpha) > 0:
	        init_alpha = self.gamma_matrix.mean(axis=0)/K
	    alpha = init_alpha.copy()
	    alphap = init_alpha.copy()
	    g_term = (psi(self.gamma_matrix) - psi(self.gamma_matrix.sum(axis=1))[:,None]).sum(axis=0)
	    for it in range(maxit):
	        grad = M *(psi(alpha.sum()) - psi(alpha)) + g_term
	        H = -M*np.diag(pg(1,alpha)) + M*pg(1,alpha.sum())
	        alpha_new = alpha - np.dot(np.linalg.inv(H),grad)
	        if (alpha_new < 0).sum() > 0:
	            init_alpha /= 10.0
	            return self.alpha_nr(maxit=maxit,init_alpha = init_alpha)
	        
	        diff = np.sum(np.abs(alpha-alpha_new))
	        alpha = alpha_new
	        if diff < 1e-6 and it > 1:
	            return alpha
	    return alpha

	# TODO: tidy up and comment this function
	def e_step(self):
		temp_beta = np.zeros((self.K,self.n_words))
		for doc in self.corpus:
			d = self.doc_index[doc]
			temp_gamma = np.zeros(self.K) + self.alpha
			for word in self.corpus[doc]:
				w = self.word_index[word]
				self.phi_matrix[doc][word] = self.beta_matrix[:,w]*np.exp(psi(self.gamma_matrix[d,:])).T
				# for k in range(self.K):
				# 	self.phi_matrix[doc][word][k] = self.beta_matrix[k,w]*np.exp(scipy.special.psi(self.gamma_matrix[d,k]))
				self.phi_matrix[doc][word] /= self.phi_matrix[doc][word].sum()
				temp_gamma += self.phi_matrix[doc][word]*self.corpus[doc][word]
				temp_beta[:,w] += self.phi_matrix[doc][word] * self.corpus[doc][word]
			# self.phi_matrix[d,:,:] = (self.beta_matrix * self.word_matrix[d,:][None,:] * (np.exp(scipy.special.psi(self.gamma_matrix[d,:]))[:,None])).T
			# self.phi_matrix[d,:,:] /= self.phi_matrix[d,:,:].sum(axis=1)[:,None]
			# self.gamma_matrix[d,:] = self.alpha + self.phi_matrix[d,:,:].sum(axis=0)
			self.gamma_matrix[d,:] = temp_gamma
		return temp_beta

	# I don't think this function is ever used....
	def m_step(self):
		for k in range(self.K):
			self.beta_matrix[k,:] = self.eta + (self.word_matrix * self.phi_matrix[:,:,k]).sum(axis=0)
		self.beta_matrix /= self.beta_matrix.sum(axis=1)[:,None]


	# Function to find the unique words in the corpus and assign them to indices
	def find_unique_words(self):
		word_index = {}
		pos = 0
		for doc in self.corpus:
			for word in self.corpus[doc]:
				if not word in word_index:
					word_index[word] = pos
					pos += 1
		print "Found {} unique words".format(len(word_index))
		return word_index

	# Pretty sure this matrix is never used
	def make_doc_index(self):
		self.doc_index = {}
		doc_pos = 0
		for doc in self.corpus:
			self.doc_index[doc] = doc_pos
			doc_pos += 1


	# Initialise the VB algorithm
	# TODO: tidy this up
	def init_vb(self):
		# self.gamma_matrix = np.zeros((self.n_docs,self.K),np.float) + 1.0
		# self.phi_matrix = np.zeros((self.n_docs,self.n_words,self.K))
		self.phi_matrix = {}
		self.gamma_matrix = np.zeros((self.n_docs,self.K))
		for doc in self.corpus:
			self.phi_matrix[doc] = {}
			for word in self.corpus[doc]:
				self.phi_matrix[doc][word] = np.zeros(self.K)
			d = self.doc_index[doc]
			doc_total = 0.0
			for word in self.corpus[doc]:
				doc_total += self.corpus[doc][word]
			self.gamma_matrix[d,:] = self.alpha + 1.0*doc_total/self.K
		# # Normalise this to sum to 1
		# self.phi_matrix /= self.phi_matrix.sum(axis=2)[:,:,None]

		# Initialise the betas
		self.beta_matrix = np.random.rand(self.K,self.n_words)
		self.beta_matrix /= self.beta_matrix.sum(axis=1)[:,None]

	# Function to return a dictionary with keys equal to documents and values equal to the probability
	# of the requested document (used for visusaling in DESI imaging)
	def get_topic_as_doc_dict(self,topic_id,thresh = 0.001,normalise=False):
		top = {}
		mat = self.gamma_matrix
		if normalise:
			mat = self.get_expect_theta()

		for doc in self.doc_index:
			pos = self.doc_index[doc]
			if mat[pos,topic_id] >= thresh:
				top[doc] = mat[pos,topic_id]
		return top

	# Return a topic as a dictionary over words
	def get_topic_as_dict(self,topic_id):
		top = {}
		for word in self.word_index:
			top[word] = self.beta_matrix[topic_id,self.word_index[word]]
		return top

	# Return the topic probabilities for all documents 
	# Note that self.doc_index maps the document names to their
	# position in this matrix
	def get_expect_theta(self):
		e_theta = self.gamma_matrix.copy()
		e_theta /= e_theta.sum(axis=1)[:,None]
		return e_theta

	def get_beta(self):
		return self.beta_matrix.copy()

# MS1 object used by Variational Bayes LDA
class MS1(object):
    def __init__(self,ms1_id,mz,rt,intensity,name):
        self.ms1_id = ms1_id
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.name = name
    def __str__(self):
        return self.name

# Some useful plotting code (uses plotly)
# Should put this into a separate file
class VariationalLDAPlotter(object):
	def __init__(self,v_lda):
		plotly.offline.init_notebook_mode()
		self.v_lda = v_lda

	def bar_alpha(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.alpha,
				)
			)
		plotly.offline.iplot({'data':data})
	def mean_gamma(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.gamma_matrix.mean(axis=0),
				)
			)
		plotly.offline.iplot({'data':data})


# TODO: comment this class!
class MultiFileVariationalLDA(object):
	def __init__(self,corpus_list,word_index,K = 20,alpha=1,eta = 0.1):
		self.word_index = word_index # this needs to be consistent across the instances
		self.corpus_list = corpus_list
		self.K = K
		self.alpha = alpha
		if type(self.alpha) == int:
			self.alpha = self.alpha*np.ones(self.K)
		self.eta = eta # Smoothing parameter for beta
		self.individual_lda = []
		for corpus in self.corpus_list:
			new_lda = VariationalLDA(corpus=corpus,K=K,alpha=alpha,eta=eta,word_index=word_index)
			self.individual_lda.append(new_lda)


	def run_vb(self,n_its = 10,initialise=True):
		if initialise:
			for l in self.individual_lda:
				l.init_vb()
		for it in range(n_its):
			print "Iteration: {}".format(it)
			temp_beta = np.zeros((self.individual_lda[0].K,self.individual_lda[0].n_words),np.float)
			total_difference = []
			for l in self.individual_lda:
				temp_beta += l.e_step()
				if l.update_alpha:
					l.alpha = l.alpha_nr()
			temp_beta /= temp_beta.sum(axis=1)[:,None]
			total_difference = (np.abs(temp_beta - self.individual_lda[0].beta_matrix)).sum()
			for l in self.individual_lda:
				l.beta_matrix = temp_beta
			print total_difference


class VariationalLDAPlotter(object):
	def __init__(self,v_lda):
		plotly.offline.init_notebook_mode()
		self.v_lda = v_lda

	def bar_alpha(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.alpha,
				)
			)
		plotly.offline.iplot({'data':data})
	def mean_gamma(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.gamma_matrix.mean(axis=0),
				)
			)
		plotly.offline.iplot({'data':data})

class MultiFileVariationalLDAPlotter(object):
	def __init__(self,m_lda):
		plotly.offline.init_notebook_mode()
		self.m_lda = m_lda

	def multi_alpha(self,normalise=False,names=None):
		data = []
		K = self.m_lda.individual_lda[0].K
		for i,l in enumerate(self.m_lda.individual_lda):
			if normalise:
				a = l.alpha / l.alpha.sum()
			else:
				a = l.alpha
			if not names == None:
				name = names[i]
			else:
				name = 'trace {}'.format(i)
			data.append(
				Bar(
					x = range(K),
					y = a,
					name = name
					)
				)
		plotly.offline.iplot({'data':data})