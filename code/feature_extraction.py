# Simon's attempts to make a single feature selection pipeline
from Queue import PriorityQueue
import numpy as np

class MS1(object):
	def __init__(self,id,mz,rt,intensity,file_name):
		self.id = id
		self.mz = mz
		self.rt = rt
		self.intensity = intensity
		self.file_name = file_name
		self.name = "{}_{}".format(self.mz,self.rt)

	def __str__(self):
		return self.name

class CorpusMaker(object):
	def __init__(self,input_type,input_set,min_loss = 10.0,
		max_loss = 200.0,fragment_tol = 7.0,loss_tol = 14.0,
		seed_words = [],min_intensity = 0.0):
		self.input_type = input_type
		self.input_set = input_set
		self.min_loss = min_loss
		self.max_loss = max_loss
		self.fragment_tol = fragment_tol
		self.loss_tol = loss_tol
		self.seed_words = seed_words
		self.min_intensity = min_intensity

		if self.input_type == 'csv':
			self.load_csv()
		if self.input_type == 'gnps':
			self.load_gnps()

		self.make_queues()
		self.make_corpus_from_queue()
		self.remove_zero_words()



	def make_queues(self):
		self.fragment_queue = PriorityQueue()
		self.loss_queue = PriorityQueue()

		if len(self.seed_words) > 0:
			for seed in self.seed_words:
				seed_mass = float(seed.split('_')[1])
				if seed.startswith('fragment'):
					self.fragment_queue.put((seed_mass,None,None,None,None,None))
				else:
					self.loss_queue.put((seed_mass,None,None,None,None,None))

		for ms2 in self.ms2:
			if ms2[2] > self.min_intensity:
				self.fragment_queue.put(ms2)
				loss_mass = ms2[3].mz - ms2[0]
				if (loss_mass > self.min_loss) & (loss_mass < self.max_loss):
					self.loss_queue.put((loss_mass,ms2[1],ms2[2],ms2[3],ms2[4],ms2[5]))


	def make_corpus_from_queue(self):
		self.corpus = {}
		for file_name in self.files:
			self.corpus[file_name] = {}
		self.n_words = 0
		self.word_counts = {}
		self.seed_counts = {}
		self.process_queue(self.fragment_queue,self.fragment_tol,'fragment')
		self.process_queue(self.loss_queue,self.loss_tol,'loss')
		print "\t Found {} words after grouping".format(self.n_words)



	def get_first_file(self):
		file_name = self.files[0]
		return self.corpus[file_name]

	def list_words(self,word_type,min_mass = 0.0,max_mass = 1000.0):
		for word in self.word_counts:
			if word.startswith(word_type):
				word_mass = float(word.split('_')[1])
				if (word_mass > min_mass and word_mass < max_mass):
					print "{}: {}".format(word,self.word_counts[word])

	def remove_zero_words(self):
		to_remove = []
		for word in self.word_counts:
			if self.word_counts[word] == 0:
				to_remove.append(word)
		for word in to_remove:
			del self.word_counts[word]
			if word in self.seed_counts:
				del self.seed_counts[word]
		print "\t Removed {} words".format(len(to_remove))
		print "\t Finished, {} words (including {} seeds)".format(len(self.word_counts),len(self.seed_counts))

	def process_queue(self,q,tolerance,prefix):
		current_item = q.get()
		current_mass = current_item[0]
		sub_list = [current_item]
		while not q.empty():
			new_item = q.get()
			new_mass = new_item[0]
			if 1e6*abs((new_mass - current_mass)/new_mass) < tolerance:
				sub_list.append(new_item)
			else:
				# this is a new group
				tot_mass = 0.0
				seeds = []
				for item in sub_list:
					if item[-1] == None:
						seeds.append(item)
					tot_mass += item[0]

				if len(seeds) == 0:
					mean_mass = tot_mass / (1.0*len(sub_list))
					word_name = prefix + "_{}".format(mean_mass)
					self.word_counts[word_name] = len(sub_list)
					self.n_words += 1
				else:
					seed_names = []
					seed_masses = []
					for seed in seeds:
						seed_names.append(prefix + "_{}".format(seed[0]))
						seed_masses.append(seed[0])
						self.word_counts[seed_names[-1]] = 0
						self.seed_counts[seed_names[-1]] = 0

					self.n_words += len(seeds)
					seed_masses = np.array(seed_masses)
				for item in sub_list:
					if not item[-1] == None:
						# It's not a seed
						file_name = item[4]
						doc_name = item[3].name
						if not doc_name in self.corpus[file_name]:
							self.corpus[file_name][doc_name] = {}

						if len(seeds) > 0:
							# Find the closest seed
							di = np.abs(item[0] - seed_masses)
							word_name = seed_names[di.argmin()]
							self.word_counts[word_name] += 1
							self.seed_counts[word_name] += 1

						if word_name in self.corpus[file_name][doc_name]:
							self.corpus[file_name][doc_name][word_name] += item[2]
						else:
							self.corpus[file_name][doc_name][word_name] = item[2]


				sub_list = [new_item]
				current_mass = new_item[0]
		if len(sub_list) > 0:
				# this is a new group
			tot_mass = 0.0
			seeds = []
			for item in sub_list:
				if item[-1] == None:
					seeds.append(item)
				tot_mass += item[0]

			if len(seeds) == 0:
				mean_mass = tot_mass / (1.0*len(sub_list))
				word_name = prefix + "_{}".format(mean_mass)
				self.word_counts[word_name] = len(sub_list)
				self.n_words += 1
			else:
				seed_names = []
				seed_masses = []
				for seed in seeds:
					seed_names.append(prefix + "_{}".format(seed[0]))
					seed_masses.append(seed[0])
					self.word_counts[seed_names[-1]] = 0
					
				self.n_words += len(seeds)
				seed_masses = np.array(seed_masses)
			for item in sub_list:
				if not item[-1] == None:
					# It's not a seed
					file_name = item[4]
					doc_name = item[3].name
					if not doc_name in self.corpus[file_name]:
						self.corpus[file_name][doc_name] = {}

					if len(seeds) > 0:
						# Find the closest seed
						di = np.abs(item[0] - seed_masses)
						word_name = seed_names[di.argmin()]
						self.word_counts[word_name] += 1

					if word_name in self.corpus[file_name][doc_name]:
						self.corpus[file_name][doc_name][word_name] += item[2]
					else:
						self.corpus[file_name][doc_name][word_name] = item[2]

	def load_gnps(self):
		self.files = ['gnps']
		self.ms1 = []
		self.ms1_index = {}
		self.ms2 = []
		self.metadata = {}
		n_processed = 0
		ms2_id = 0
		self.metadata = {}
		for file in self.input_set:
			with open(file,'r') as f:
				doc_name = file.split('/')[-1]
				self.metadata[doc_name] = {}
				new_ms1 = MS1(str(n_processed),None,None,None,'gnps')
				new_ms1.name = doc_name
				self.ms1.append(new_ms1)
				self.ms1_index[str(n_processed)] = new_ms1
				for line in f:
					rline = line.rstrip()
					if len(rline) > 0:
						if rline.startswith('>'):
							keyval = rline[1:].split(' ')[0]
							valval = rline[len(keyval)+2:]
							if not keyval == 'ms2peaks':
								self.metadata[doc_name][keyval] = valval
							if keyval == 'compound':
								self.metadata[doc_name]['annotation'] = valval
							if keyval == 'parentmass':
								self.ms1[-1].mz = float(valval)
							if keyval == 'intensity':
								self.ms1[-1].intensity = float(valval)
						else:
							# If it gets here, its a fragment peak
							sr = rline.split(' ')
							mass = float(sr[0])
							intensity = float(sr[1])
							parent = self.ms1[-1]
							if intensity > self.min_intensity:
								new_ms2 = (mass,0.0,intensity,parent,'gnps',float(ms2_id))
								self.ms2.append(new_ms2)
								ms2_id += 1
				n_processed += 1
			if n_processed % 100 == 0:
				print "Processed {} spectra".format(n_processed)



	def load_csv(self):
		# input set is a list of tuples, each one has ms1 and ms2 files
		# Load the ms1 files
		self.files = []
		self.ms1 = []
		self.ms1_index = {}
		self.ms2 = []
		for input in self.input_set:
			file_name = input[0].split('/')[-1].split('_ms1')[0]
			print file_name
			self.files.append(file_name)
			with open(input[0],'r') as f:
				heads = f.readline().split(',') # remove headings
				rt_pos = heads.index('"rt"')
				mz_pos = heads.index('"mz"')
				intensity_pos = heads.index('"intensity"')
				peak_id_pos = heads.index('"peakID"')
				for line in f:
					tokens = line.split(',')
					rt = float(tokens[rt_pos])
					mz = float(tokens[mz_pos])
					intensity = float(tokens[intensity_pos])
					id = tokens[peak_id_pos]
					new_ms1 = MS1(id,mz,rt,intensity,file_name)
					self.ms1.append(new_ms1)
					self.ms1_index[id] = new_ms1
			print "\t loaded {} ms1 peaks".format(len(self.ms1))

			with open(input[1],'r') as f:
				heads = f.readline().split(',')
				rt_pos = heads.index('"rt"')
				mz_pos = heads.index('"mz"')
				intensity_pos = heads.index('"intensity"')
				peak_id_pos = heads.index('"peakID"')
				parent_id_pos = heads.index('"MSnParentPeakID"')
				for line in f:
					tokens = line.split(',')
					rt = float(tokens[rt_pos])
					mz = float(tokens[mz_pos])
					intensity = float(tokens[intensity_pos])					
					id = tokens[peak_id_pos]
					parent_id = tokens[parent_id_pos]
					parent = self.ms1_index[parent_id]
					self.ms2.append((mz,rt,intensity,parent,file_name,id))
				
				print "\t loaded {} ms2 peaks".format(len(self.ms2))

					


