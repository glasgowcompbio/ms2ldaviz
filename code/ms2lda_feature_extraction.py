# Simon's attempts to make a single feature selection pipeline
from Queue import PriorityQueue
import numpy as np
import sys
# sys.path.append('/Users/simon/git/efcompute')
# from ef_assigner import ef_assigner
# from formula import Formula
# from ef_constants import ATOM_MASSES, PROTON_MASS, ATOM_NAME_LIST

# Restructuring in December 2016
# Feature selection has three steps:
#  1. Loading a bunch of spectra into some standard format
#  2. Turning them into fragment and loss features
#  3. Making the corpus object

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

# Abstract loader class
class Loader(object):
    def load_spectra(self,input_set):
        raise NotImplementedError("load spectra method must be implemented")


# A class to load Emma's data
# This is very rough
class LoadEmma(Loader):
    def __init__(self,min_intensity = 0.0):
        self.min_intensity = min_intensity
    def __str__(self):
        return "Loader for Emma data format"
    def load_spectra(self,input_set):
        import pymzml

        ms1 = []
        ms2 = []
        metadata = {}
        nspec = 0
        nc = 0
        ms2_id = 0

        for input_file in input_set:

            # Load the mzml object
            run = pymzml.run.Reader(input_file, MS1_Precision=5e-6)


            file_name = input_file.split('/')[-1]

            for spectrum in run:
                if 'collision-induced dissociation' in spectrum:
                    parentmass = spectrum['precursors'][0]['mz']
                    parentrt = spectrum['scan start time']
                    nc += 1
                    newms1 = MS1(nc,parentmass,parentrt,None,file_name)
                    ms1.append(newms1)
                    metadata[newms1.name] = {'parentmass':parentmass,'parentrt':parentrt}
                    for mz,intensity in spectrum.peaks:
                        if intensity > self.min_intensity:
                            ms2.append((mz,parentrt,intensity,ms1[-1],file_name,float(ms2_id)))
                            ms2_id += 1
                elif 'beam-type collision-induced dissociation' in spectrum:
                    pass
                else:
                    pass
                nspec += 1

        return ms1,ms2,metadata

# A class to load GNPS / MASSBANK style CSI spectra that treats the different
# collision energies as different documents
class LoadGNPSSeperateCollisions(Loader):
    def __init__(self,min_intensity = 0.0):
        self.min_intensity = min_intensity

    def load_spectra(self,input_set):
        # Define dummy files object
        files = ['gnps']
        self.ms1 = []
        self.ms2 = []
        self.metadata = {}
        ms1_id = 0
        ms2_id = 0
        for input_file in input_set:
            with open(input_file,'r') as f:
                in_doc = False
                doc_name_prefix = input_file.split('/')[-1]
                temp_metadata = {}
                current_collision = None
                temp_ms2 = []
                for line in f:
                    rline = line.rstrip()
                    if len(rline) > 0:
                        if rline.startswith('>'):
                            # This is some kind of metadata
                            if not rline.startswith('>collision'):
                                keyval = rline[1:].split()
                                key = keyval[0]
                                val = keyval[1]
                                temp_metadata[key] = val
                                if key == 'compound':
                                    temp_metadata['annotation'] = val
                                elif key == 'parentmass':
                                    temp_metadata['parentmass'] = float(val)
                            else:
                                # This is a new peak block that should be made into a new document
                                
                                if in_doc:
                                    # If we are in a document, we need to save it
                                    doc_name = doc_name_prefix + '_collision_' + current_collision
                                    temp_metadata['collision'] = current_collision
                                    self.metadata[doc_name] = temp_metadata.copy()                            
                                    new_ms1 = MS1(ms1_id,temp_metadata['parentmass'],None,None,'gnps')
                                    new_ms1.name = doc_name
                                    ms1_id += 1
                                    self.ms1.append(new_ms1)
                                    for peak in temp_ms2:
                                        self.ms2.append((peak[0],0.0,peak[1],new_ms1,'gnps',float(ms2_id)))
                                        ms2_id += 1
                                temp_ms2 = []
                                in_doc = True
                                current_collision = rline.split()[1]
                        else:
                            # its a peak
                            mzi = rline.split()
                            temp_ms2.append((float(mzi[0]),float(mzi[1])))
                # Got to the end of the file
                if len(temp_ms2) > 0:
                    # We have a document to make
                    doc_name = doc_name_prefix + '_collision_' + current_collision
                    new_ms1 = MS1(ms1_id,temp_metadata['parentmass'],None,None,'gnps')
                    self.ms1.append(new_ms1)
                    temp_metadata['collision'] = current_collision
                    self.metadata[doc_name] = temp_metadata.copy()
                    new_ms1.name = doc_name
                    for peak in temp_ms2:
                        self.ms2.append((peak[0],0.0,peak[1],new_ms1,'gnps',float(ms2_id)))
                        ms2_id += 1
        return self.ms1,self.ms2,self.metadata


class LoadGNPS(Loader):
    def __init__(self,merge_energies = True,merge_ppm = 2,replace = 'sum',min_intensity = 0.0):
        self.merge_energies = merge_energies
        self.merge_ppm = merge_ppm
        self.replace = replace
        self.min_intensity = min_intensity
    def load_spectra(self,input_set):
        # Input set is a list of files, one for each spectra
        self.input_set = input_set
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
                temp_mass = []
                temp_intensity = []
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
                            if intensity > self.min_intensity:
                                if self.merge_energies and len(temp_mass)>0:
                                    errs = 1e6*np.abs(mass-np.array(temp_mass))/mass
                                    if errs.min() < self.merge_ppm:
                                        # Don't add, but merge the intensity
                                        min_pos = errs.argmin()
                                        if self.replace == 'max':
                                            temp_intensity[min_pos] = max(intensity,temp_intensity[min_pos])
                                        else:
                                            temp_intensity[min_pos] += intensity
                                    else:
                                        temp_mass.append(mass)
                                        temp_intensity.append(intensity)
                                else:
                                    temp_mass.append(mass)
                                    temp_intensity.append(intensity)

                parent = self.ms1[-1]
                for mass,intensity in zip(temp_mass,temp_intensity):
                    new_ms2 = (mass,0.0,intensity,parent,'gnps',float(ms2_id))
                    self.ms2.append(new_ms2)
                    ms2_id += 1

                n_processed += 1
            if n_processed % 100 == 0:
                print "Processed {} spectra".format(n_processed)
        return self.ms1,self.ms2,self.metadata

# A class to load spectra that sit in MSP files
class LoadMSP(Loader):
    def load_spectra(self,input_set):
        ms1 = []
        ms2 = []
        metadata = {}
        ms2_id = 0
        ms1_id = 0
        for input_file in input_set:
            file_name = input_file.split('/')[-1]
            with open(input_file,'r') as f:
                temp_metadata = {}
                in_doc = False
                parentmass = None
                parentrt = None
                for line in f:
                    rline  = line.rstrip()
                    if not in_doc and len(rline) == 0:
                        continue
                    elif in_doc and len(rline) == 0:
                        # finished doc, time to save
                        in_doc = False
                        temp_metadata = {}
                        parentmass = None
                        parentrt = None
                        new_ms1 = None
                    else:
                        tokens = rline.split()
                        if in_doc:
                            mz = float(tokens[0])
                            intensity = float(tokens[1])
                            ms2.append((mz,0.0,intensity,new_ms1,file_name,float(ms2_id)))
                            ms2_id += 1
                        elif rline.startswith('Num Peaks'):
                            in_doc = True
                            new_ms1 = MS1(ms1_id,parentmass,parentrt,None,file_name)
                            ms1_id += 1
                            doc_name = 'document_{}'.format(ms1_id)
                            metadata[doc_name] = temp_metadata.copy()
                            new_ms1.name = doc_name
                            ms1.append(new_ms1)
                        else:
                            key = tokens[0][:-1].lower()
                            val = tokens[1]
                            if key == 'precursormz':
                                temp_metadata['parentmass'] = float(val)
                                parentmass = float(val)
                            elif key == 'retentiontime':
                                temp_metadata['parentrt'] = float(val)
                                parentrt = float(val)
                            else:
                                temp_metadata[key] = val
        return ms1,ms2,metadata





# Abstract feature making class
class MakeFeatures(object):
    def make_features(self,ms2):
        raise NotImplementedError("make features method must be implemented")

class MakeNominalFeatures(MakeFeatures):
    def __init__(self,min_frag = 0.0,max_frag = 10000.0,
                      min_loss = 10.0,max_loss = 200.0,
                      min_intensity = 0.0,bin_width = 0.3):
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity
        self.bin_width = bin_width

    def __str__(self):
        return "Nominal feature extractor, bin_width = {}".format(self.bin_width)

    def __unicode__(self):
        return "Nominal feature extractor, bin_width = {}".format(self.bin_width)

    def make_features(self,ms2):
        # Just make integers between min and max values and assign to corpus
        word_names = []
        word_mz_range = {}
        self.corpus = {}
        for frag in ms2:
            parentmass = frag[3].mz
            frag_mass = frag[0]
            loss_mass = parentmass - frag_mass
            intensity = frag[2]
            doc_name = frag[3].name
            file_name = frag[4]
            if intensity >= self.min_intensity:
                if frag_mass >= self.min_frag and frag_mass <= self.max_frag:
                    frag_word = round(frag_mass)
                    err = abs(frag_mass - frag_word)
                    if err <= self.bin_width:
                        # Keep it
                        word_name = 'fragment_' + str(frag_word)
                        if not word_name in word_names:
                            word_names.append(word_name)
                            word_mz_range[word_name] = (frag_word - self.bin_width,frag_word + self.bin_width)
                        self._add_word_to_corpus(word_name,file_name,doc_name,intensity)
                if loss_mass >= self.min_loss and loss_mass <= self.max_loss:
                    loss_word = round(loss_mass)
                    err = abs(loss_mass - loss_word)
                    if err <= self.bin_width:
                        word_name = 'loss_' + str(loss_word)
                        if not word_name in word_names:
                            word_names.append(word_name)
                            word_mz_range[word_name] = (loss_word - self.bin_width,loss_word + self.bin_width)
                        self._add_word_to_corpus(word_name,file_name,doc_name,intensity)
        return self.corpus,word_mz_range

    def _add_word_to_corpus(self,word_name,file_name,doc_name,intensity):
        if not file_name in self.corpus:
            self.corpus[file_name] = {}
        if not doc_name in self.corpus[file_name]:
            self.corpus[file_name][doc_name] = {}
        if not word_name in self.corpus[file_name][doc_name]:
            self.corpus[file_name][doc_name][word_name] = intensity
        else:
            self.corpus[file_name][doc_name][word_name] += intensity

# Class to use kde to make features. Could do with tidying

class MakeKDEFeatures(MakeFeatures):
    def __init__(self,loss_ppm = 15.0,frag_ppm = 7.0,
                      min_frag = 0.0,max_frag = 10000.0,
                      min_loss = 10.0,max_loss = 200.0,
                      min_intensity = 0.0):
        self.loss_ppm = loss_ppm
        self.frag_ppm = frag_ppm
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity

        # legacy code: adds up intensities if the same feature appears multiple times
        self.replace = 'sum'

    def make_features(self,ms2):
        self.fragment_queue,self.loss_queue = self._make_queues(ms2)
        self._make_kde_corpus()
        return self.corpusesus,self.word_mz_range



    def _make_kde_corpus(self):
        # Process the losses
        loss_masses = []
        self.loss_meta = []
        while not self.loss_queue.empty():
            newitem = self.loss_queue.get()
            self.loss_meta.append(newitem)
            loss_masses.append(newitem[0])

        self.loss_array = np.array(loss_masses)


        frag_masses = []
        self.frag_meta = []
        while not self.fragment_queue.empty():
            newitem = self.fragment_queue.get()
            self.frag_meta.append(newitem)
            frag_masses.append(newitem[0])

        self.frag_array = np.array(frag_masses)

        # Set the kde paramaters
        stop_thresh = 5
        
        self.corpus = {}
        self.n_words = 0
        self.word_counts = {}
        self.word_mz_range = {}
        self.loss_kde = self._make_kde(self.loss_array,self.loss_ppm,stop_thresh)
        self.loss_groups,self.loss_group_list = self._process_kde(self.loss_kde,self.loss_array,self.loss_ppm)
        self._update_corpus(self.loss_array,self.loss_kde,self.loss_meta,self.loss_groups,self.loss_group_list,'loss')


        print "Finished losses, total words now: {}".format(len(self.word_counts))

        self.frag_kde = self._make_kde(self.frag_array,self.frag_ppm,stop_thresh)
        self.frag_groups,self.frag_group_list = self._process_kde(self.frag_kde,self.frag_array,self.frag_ppm)
        self._update_corpus(self.frag_array,self.frag_kde,self.frag_meta,self.frag_groups,self.frag_group_list,'fragment')



        print "Finished fragments, total words now: {}".format(len(self.word_counts))


    def _make_kde(self,mass_array,ppm,stop_thresh):
        kde = np.zeros_like(mass_array)
        n_losses = len(mass_array)
        for i in range(n_losses):
            if i % 1000 == 0:
                print "Done kde for {} of {}".format(i,n_losses)
            this_mass = mass_array[i]
            ss = ((ppm*(this_mass/1e6))/3.0)**2
            finished = False
            pos = i-1
            while (not finished) and (pos >= 0):
                de = self.gauss_pdf(mass_array[pos],this_mass,ss)
                kde[pos] += de
                if this_mass - mass_array[pos] > stop_thresh*np.sqrt(ss):
                    finished = True
                pos -= 1
            finished = False
            pos = i
            while (not finished) and (pos < len(mass_array)):
                de = self.gauss_pdf(mass_array[pos],this_mass,ss)
                kde[pos] += de
                if mass_array[pos] - this_mass > stop_thresh*np.sqrt(ss):
                    finished = True
                pos += 1
        print "Made kde"
        return kde


    def gauss_pdf(self,x,m,ss):
        return (1.0/np.sqrt(2*np.pi*ss))*np.exp(-0.5*((x-m)**2)/ss)

    def _process_kde(self,kde,masses,ppm):

        kde_copy = kde.copy()
        groups = np.zeros_like(kde) - 1

        max_width = 50 # ppm
        group_list = []
        current_group = 0
        # group_formulas = []
        # group_mz = []
        verbose = False
        while True:
            biggest_pos = kde_copy.argmax()
            peak_values = [biggest_pos]
            this_mass = masses[biggest_pos]
            ss = ((ppm*(this_mass/1e6))/3.0)**2
            min_val = 1.0/(np.sqrt(2*np.pi*ss))

            pos = biggest_pos
            intensity = kde_copy[biggest_pos]
            if intensity < 0.8*min_val:
                break # finished
            finished = False
            lowest_intensity = intensity
            lowest_index = biggest_pos

            if intensity > min_val:
                # Checks that it's not a singleton
                if pos > 0:
                    # Check that it's not the last one
                    while not finished:
                        pos -= 1
                        # Decide if this one should be part of the peak or not
                        if kde[pos] > lowest_intensity + 0.01*intensity or kde[pos] <= 1.001*min_val:
                            # It shouldn't
                            finished = True
                        elif groups[pos] > -1:
                            # We've hit another group!
                            finished = True
                        elif 1e6*abs(masses[pos] - this_mass)/this_mass > max_width:
                            # Gone too far
                            finished = True
                        else:
                            # it should
                            peak_values.append(pos)
                        # If it's the last one, we're finished
                        if pos == 0:
                            finished = True

                        # If it's the lowest we've seen so far remember that
                        if kde[pos] < lowest_intensity:
                            lowest_intensity = kde[pos]
                            lowest_index = pos


                pos = biggest_pos
                finished = False
                lowest_intensity = intensity
                lowest_index = biggest_pos
                if pos < len(kde)-1:
                    while not finished:
                        # Move to the right
                        pos += 1
                        if verbose:
                            print pos
                        # check if this one should be in the peak
                        if kde[pos] > lowest_intensity + 0.01*intensity or kde[pos] <= 1.001*min_val:
                            # it shouldn't
                            finished = True
                        elif groups[pos] > -1:
                            # We've hit another group!
                            finished = True
                        elif 1e6*abs(masses[pos] - this_mass)/this_mass > max_width:
                            # Gone too far
                            finished = True
                        else:
                            peak_values.append(pos)

                        # If it's the last value, we're finished
                        if pos == len(kde)-1:
                            finished = True

                        # If it's the lowest we've seen, remember that
                        if kde[pos] < lowest_intensity:
                            lowest_intensity = kde[pos]
                            lowest_index = pos

            else:
                # Singleton peak
                peak_values = [biggest_pos]
                
            if len(peak_values) > 0:
                kde_copy[peak_values] = 0.0
                groups[peak_values] = current_group
                group_id = current_group
                current_group += 1
                group_mz = masses[biggest_pos]

                # Find formulas
                hit_string = None

                new_group = (group_id,group_mz,hit_string,biggest_pos)

                group_list.append(new_group)
                if current_group % 100 == 0:
                    print "Found {} groups".format(current_group)

        return groups,group_list

    def _update_corpus(self,masses,kde,meta,groups,group_list,prefix):
        # Loop over the groups
        for group in group_list:
            group_id = group[0]
            group_mz = group[1]
            group_formula = group[2]
            pos = np.where(groups == group_id)[0]
            min_mz = 100000.0
            max_mz = 0.0
            if len(pos) > 0:
                feature_name = str(group_mz)
                feature_name = prefix + '_' + feature_name
                if group_formula:
                    feature_name += '_' + group_formula

                # (mass,0.0,intensity,parent,'gnps',float(ms2_id))

                for p in pos:
                    this_meta = meta[p]
                    this_mz = this_meta[0]
                    if this_mz <= min_mz:
                        min_mz = this_mz
                    if this_mz >= max_mz:
                        max_mz = this_mz
                    intensity = this_meta[2]
                    doc_name = this_meta[3].name
                    this_file = this_meta[4]
                    if not this_file in self.corpus:
                        self.corpus[this_file] = {}
                    if not doc_name in self.corpus[this_file]:
                        self.corpus[this_file][doc_name] = {}
                    if not feature_name in self.corpus[this_file][doc_name]:
                        self.corpus[this_file][doc_name][feature_name] = 0.0
                    if self.replace == 'sum':
                        self.corpus[this_file][doc_name][feature_name] += intensity
                    else:
                        current = self.corpus[this_file][doc_name][feature_name]
                        newval = max(current,intensity)
                        self.corpus[this_file][doc_name][feature_name] = newval

                self.word_counts[feature_name] = len(pos)
                self.word_mz_range[feature_name] = (min_mz,max_mz)
                

    def _make_queues(self,ms2):

        fragment_queue = PriorityQueue()
        loss_queue = PriorityQueue()

        for peak in ms2:
            frag_mass = peak[0]
            frag_intensity = peak[2]
            parent_mass = peak[3].mz
            if frag_intensity > self.min_intensity and frag_mass >= self.min_frag and frag_mass <= self.max_frag:
                fragment_queue.put(peak)
            loss_mass = parent_mass - frag_mass
            if frag_intensity > self.min_intensity and loss_mass >= self.min_loss and loss_mass <= self.max_loss:
                new_peak = (loss_mass,peak[1],peak[2],peak[3],peak[4],peak[5])
                loss_queue.put(new_peak)

        return fragment_queue,loss_queue

class MakeQueueFeatures(MakeFeatures):
    def __init__(self,min_frag = 0.0,max_frag = 10000.0,
                      min_loss = 10.0,max_loss = 200.0,
                      min_intensity = 0.0,gap_ppm = 7.0):
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity
        self.gap_ppm = gap_ppm

    def make_features(self,ms2):
        fragment_queue,loss_queue = self._make_queues(ms2)
        self.corpus = {}
        self.word_mz_range = {}
        self._add_queue_to_corpus(fragment_queue,'fragment')
        self._add_queue_to_corpus(loss_queue,'loss')
        return self.corpus,self.word_mz_range

    def _add_queue_to_corpus(self,queue,name_prefix):
        current_item = queue.get()
        temp_peaks = [current_item]
        current_mz = current_item[0]
        while not queue.empty():
            new_item = queue.get()
            new_mz = new_item[0]
            # Check if we have hit a gap
            if 1e6*abs(new_mz-current_mz)/current_mz < self.gap_ppm:
                # Still in the same feature
                current_mz = new_mz
                temp_peaks.append(new_item)
            else:
                # Weve found a gap, so process the ones weve collected
                # Calculate the average mass
                masses = [p[0] for p in temp_peaks]
                tot_mass = sum(masses)
                min_mass = min(masses)
                max_mass = max(masses)

                mean_mass = tot_mass/(1.0*len(temp_peaks))
                word_name = name_prefix + '_{}'.format(mean_mass)

                for peak in temp_peaks:
                    file_name = peak[4]
                    doc_name = peak[3].name
                    intensity = peak[2]
                    if not file_name in self.corpus:
                        self.corpus[file_name] = {}
                    if not doc_name in self.corpus[file_name]:
                        self.corpus[file_name][doc_name] = {}
                    if not word_name in self.corpus[file_name][doc_name]:
                        self.corpus[file_name][doc_name][word_name] = intensity
                    else:
                        self.corpus[file_name][doc_name][word_name] += intensity

                self.word_mz_range[word_name] = (min_mass,max_mass)
                
                temp_peaks = [new_item]
                current_mz = new_mz


    def _make_queues(self,ms2):

        fragment_queue = PriorityQueue()
        loss_queue = PriorityQueue()

        for peak in ms2:
            frag_mass = peak[0]
            frag_intensity = peak[2]
            parent_mass = peak[3].mz
            if frag_intensity > self.min_intensity and frag_mass >= self.min_frag and frag_mass <= self.max_frag:
                fragment_queue.put(peak)
            loss_mass = parent_mass - frag_mass
            if frag_intensity > self.min_intensity and loss_mass >= self.min_loss and loss_mass <= self.max_loss:
                new_peak = (loss_mass,peak[1],peak[2],peak[3],peak[4],peak[5])
                loss_queue.put(new_peak)

        return fragment_queue,loss_queue


# A utility function that takes a corpus object and returns a numpy matrix and two index objects
# the corpus should be a single corpus, and NOT a list of corpuses
def corpusToMatrix(corpus,word_list,just_fragments = True):
    # Make the indices
    doc_index = {}
    word_index = {}
    pos = 0
    for doc in corpus:
        doc_index[doc] = pos
        pos += 1
    pos = 0
    for word in word_list:
        if just_fragments and word.startswith('loss'):
            continue
        word_index[word] = pos
        pos += 1
    
    import numpy as np

    mat = np.zeros((len(doc_index),len(word_index)))
    for doc in corpus:
        doc_pos = doc_index[doc]
        for word in corpus[doc]:
            if word in word_index:
                word_pos = word_index[word]
                mat[doc_pos,word_pos] = corpus[doc][word]

    return mat,doc_index,word_index


class MS2LDAFeatureExtractor(object):
    def __init__(self,input_set,loader,feature_maker):
        self.input_set = input_set
        self.loader = loader
        print self.loader
        self.feature_maker = feature_maker
        print self.feature_maker

        print "Loading spectra"
        self.ms1,self.ms2,self.metadata = self.loader.load_spectra(self.input_set)
        print "Creating corpus"
        self.corpus,self.word_mz_range = self.feature_maker.make_features(self.ms2)




# class CorpusMaker(object):
#     def __init__(self,input_type,input_set,min_loss = 10.0,
#         max_loss = 200.0,fragment_tol = 7.0,loss_tol = 14.0,
#         seed_words = [],min_intensity = 0.0,algorithm = 'kde',
#         replace = 'max',formulas = None):
#         self.replace = replace
#         self.input_type = input_type
#         self.input_set = input_set
#         self.min_loss = min_loss
#         self.max_loss = max_loss
#         self.fragment_tol = fragment_tol
#         self.loss_tol = loss_tol
#         self.seed_words = seed_words
#         self.min_intensity = min_intensity
#         self.formulas = formulas

#         if self.input_type == 'csv':
#             self.load_csv()
#         if self.input_type == 'gnps':
#             self.load_gnps()
#         if self.input_type == 'metfamily':
#             self.load_metfamily_matrix()
#             # Loads corpus directly, so can just return
#             return 

#         self.make_queues()
            
#         if algorithm == 'queue':    
#             self.make_corpus_from_queue()
#             self.remove_zero_words()
#         elif algorithm == 'kde':
#             self.make_kde_corpus()


#     def make_kde_corpus(self,loss_ppm = 15.0,frag_ppm = 7.0):
#         # Process the losses
#         loss_masses = []
#         self.loss_meta = []
#         while not self.loss_queue.empty():
#             newitem = self.loss_queue.get()
#             self.loss_meta.append(newitem)
#             loss_masses.append(newitem[0])

#         self.loss_array = np.array(loss_masses)


#         frag_masses = []
#         self.frag_meta = []
#         while not self.fragment_queue.empty():
#             newitem = self.fragment_queue.get()
#             self.frag_meta.append(newitem)
#             frag_masses.append(newitem[0])

#         self.frag_array = np.array(frag_masses)

#         # Set the kde paramaters
#         stop_thresh = 5
        
#         self.corpus = {}
#         for file_name in self.files:
#             self.corpus[file_name] = {}
#         self.n_words = 0
#         self.word_counts = {}
#         self.word_mz_range = {}
#         self.loss_kde = self.make_kde(self.loss_array,loss_ppm,stop_thresh)
#         self.loss_groups,self.loss_group_list = self.process_kde(self.loss_kde,self.loss_array,loss_ppm,ef_polarity = 'none',formulas = self.formulas)
#         self.update_corpus(self.loss_array,self.loss_kde,self.loss_meta,self.loss_groups,self.loss_group_list,'loss')


#         print "Finished losses, total words now: {}".format(len(self.word_counts))

#         self.frag_kde = self.make_kde(self.frag_array,frag_ppm,stop_thresh)
#         self.frag_groups,self.frag_group_list = self.process_kde(self.frag_kde,self.frag_array,frag_ppm,ef_polarity = 'pos',formulas = self.formulas)
#         self.update_corpus(self.frag_array,self.frag_kde,self.frag_meta,self.frag_groups,self.frag_group_list,'fragment')



#         print "Finished fragments, total words now: {}".format(len(self.word_counts))


#     def update_corpus(self,masses,kde,meta,groups,group_list,prefix):
#         # Loop over the groups
#         for group in group_list:
#             group_id = group[0]
#             group_mz = group[1]
#             group_formula = group[2]
#             pos = np.where(groups == group_id)[0]
#             min_mz = 100000.0
#             max_mz = 0.0
#             if len(pos) > 0:
#                 feature_name = str(group_mz)
#                 feature_name = prefix + '_' + feature_name
#                 if group_formula:
#                     feature_name += '_' + group_formula

#                 # (mass,0.0,intensity,parent,'gnps',float(ms2_id))

#                 for p in pos:
#                     this_meta = meta[p]
#                     this_mz = this_meta[0]
#                     if this_mz <= min_mz:
#                         min_mz = this_mz
#                     if this_mz >= max_mz:
#                         max_mz = this_mz
#                     intensity = this_meta[2]
#                     doc_name = this_meta[3].name
#                     this_file = this_meta[4]
#                     if not doc_name in self.corpus[this_file]:
#                         self.corpus[this_file][doc_name] = {}
#                     if not feature_name in self.corpus[this_file][doc_name]:
#                         self.corpus[this_file][doc_name][feature_name] = 0.0
#                     if self.replace == 'sum':
#                         self.corpus[this_file][doc_name][feature_name] += intensity
#                     else:
#                         current = self.corpus[this_file][doc_name][feature_name]
#                         newval = max(current,intensity)
#                         self.corpus[this_file][doc_name][feature_name] = newval

#                 self.word_counts[feature_name] = len(pos)
#                 self.word_mz_range[feature_name] = (min_mz,max_mz)
                

#     def process_kde(self,kde,masses,ppm,ef_polarity,formulas = None,formula_ppm = 10):

#         if formulas:
#             from formula import Formula
#             # User has supplied a list of formulas to match
#             formula_objs = []
#             formula_masses = []
#             for formula in formulas:
#                 formula_objs.append(Formula(formula))
#                 formula_masses.append(formula_objs[-1].compute_exact_mass())
#             formula_masses = np.array(formula_masses)


#         atom_list = ['H','C','N','O']
#         atom_list2 = ['H','C','N','O','P']
#         print "Performing kernel density estimation"
#         ef7 = ef_assigner(do_7_rules = True,verbose = False,atom_list = atom_list)
#         ef72 = ef_assigner(do_7_rules = True,verbose = False,atom_list = atom_list2)
#         ef3 = ef_assigner(do_7_rules = False,verbose = False,atom_list = atom_list2)

#         kde_copy = kde.copy()
#         groups = np.zeros_like(kde) - 1

#         max_width = 50 # ppm
#         group_list = []
#         current_group = 0
#         # group_formulas = []
#         # group_mz = []
#         verbose = False
#         while True:
#             biggest_pos = kde_copy.argmax()
#             peak_values = [biggest_pos]
#             this_mass = masses[biggest_pos]
#             ss = ((ppm*(this_mass/1e6))/3.0)**2
#             min_val = 1.0/(np.sqrt(2*np.pi*ss))

#             pos = biggest_pos
#             intensity = kde_copy[biggest_pos]
#             if intensity < 0.8*min_val:
#                 break # finished
#             finished = False
#             lowest_intensity = intensity
#             lowest_index = biggest_pos

#             if intensity > min_val:
#                 # Checks that it's not a singleton
#                 if pos > 0:
#                     # Check that it's not the last one
#                     while not finished:
#                         pos -= 1
#                         # Decide if this one should be part of the peak or not
#                         if kde[pos] > lowest_intensity + 0.01*intensity or kde[pos] <= 1.001*min_val:
#                             # It shouldn't
#                             finished = True
#                         elif groups[pos] > -1:
#                             # We've hit another group!
#                             finished = True
#                         elif 1e6*abs(masses[pos] - this_mass)/this_mass > max_width:
#                             # Gone too far
#                             finished = True
#                         else:
#                             # it should
#                             peak_values.append(pos)
#                         # If it's the last one, we're finished
#                         if pos == 0:
#                             finished = True

#                         # If it's the lowest we've seen so far remember that
#                         if kde[pos] < lowest_intensity:
#                             lowest_intensity = kde[pos]
#                             lowest_index = pos


#                 pos = biggest_pos
#                 finished = False
#                 lowest_intensity = intensity
#                 lowest_index = biggest_pos
#                 if pos < len(kde)-1:
#                     while not finished:
#                         # Move to the right
#                         pos += 1
#                         if verbose:
#                             print pos
#                         # check if this one should be in the peak
#                         if kde[pos] > lowest_intensity + 0.01*intensity or kde[pos] <= 1.001*min_val:
#                             # it shouldn't
#                             finished = True
#                         elif groups[pos] > -1:
#                             # We've hit another group!
#                             finished = True
#                         elif 1e6*abs(masses[pos] - this_mass)/this_mass > max_width:
#                             # Gone too far
#                             finished = True
#                         else:
#                             peak_values.append(pos)

#                         # If it's the last value, we're finished
#                         if pos == len(kde)-1:
#                             finished = True

#                         # If it's the lowest we've seen, remember that
#                         if kde[pos] < lowest_intensity:
#                             lowest_intensity = kde[pos]
#                             lowest_index = pos

#             else:
#                 # Singleton peak
#                 peak_values = [biggest_pos]
                
#             if len(peak_values) > 0:
#                 kde_copy[peak_values] = 0.0
#                 groups[peak_values] = current_group
#                 group_id = current_group
#                 current_group += 1
#                 group_mz = masses[biggest_pos]

#                 # Find formulas
#                 hit_string = None
#                 if formulas:
#                     errs = 1e6*np.abs(group_mz - formula_masses)/group_mz
#                     if errs.min() < formula_ppm:
#                         hit_string = str(formula_objs[errs.argmin()])
#                 else:
#                     form,hit_string,mass = ef7.find_formulas([masses[biggest_pos]],ppm=formula_ppm,polarisation=ef_polarity)
#                     hit_string = hit_string[0]
#                     if hit_string == None:
#                         form,hit_string,mass = ef72.find_formulas([masses[biggest_pos]],ppm=formula_ppm,polarisation=ef_polarity)
#                         hit_string = hit_string[0]
#                         if hit_string == None:
#                             form,hit_string,mass = ef3.find_formulas([masses[biggest_pos]],ppm=formula_ppm,polarisation=ef_polarity)
#                             hit_string = hit_string[0]

#                     if not hit_string == None:
#                         hit_string = '?' + hit_string

#                 new_group = (group_id,group_mz,hit_string,biggest_pos)

#                 group_list.append(new_group)
#                 if current_group % 100 == 0:
#                     print "Found {} groups".format(current_group)

#         return groups,group_list


  
#     def make_kde(self,mass_array,ppm,stop_thresh):
#         kde = np.zeros_like(mass_array)
#         n_losses = len(mass_array)
#         for i in range(n_losses):
#             if i % 1000 == 0:
#                 print "Done kde for {} of {}".format(i,n_losses)
#             this_mass = mass_array[i]
#             ss = ((ppm*(this_mass/1e6))/3.0)**2
#             finished = False
#             pos = i-1
#             while (not finished) and (pos >= 0):
#                 de = self.gauss_pdf(mass_array[pos],this_mass,ss)
#                 kde[pos] += de
#                 if this_mass - mass_array[pos] > stop_thresh*np.sqrt(ss):
#                     finished = True
#                 pos -= 1
#             finished = False
#             pos = i
#             while (not finished) and (pos < len(mass_array)):
#                 de = self.gauss_pdf(mass_array[pos],this_mass,ss)
#                 kde[pos] += de
#                 if mass_array[pos] - this_mass > stop_thresh*np.sqrt(ss):
#                     finished = True
#                 pos += 1
#         print "Made kde"
#         return kde


#     def gauss_pdf(self,x,m,ss):
#         return (1.0/np.sqrt(2*np.pi*ss))*np.exp(-0.5*((x-m)**2)/ss)

#     def make_queues(self):
#         self.fragment_queue = PriorityQueue()
#         self.loss_queue = PriorityQueue()

#         if len(self.seed_words) > 0:
#             for seed in self.seed_words:
#                 seed_mass = float(seed.split('_')[1])
#                 if seed.startswith('fragment'):
#                     self.fragment_queue.put((seed_mass,None,None,None,None,None))
#                 else:
#                     self.loss_queue.put((seed_mass,None,None,None,None,None))

#         for ms2 in self.ms2:
#             if ms2[2] > self.min_intensity:
#                 self.fragment_queue.put(ms2)
#                 loss_mass = ms2[3].mz - ms2[0]
#                 if (loss_mass > self.min_loss) & (loss_mass < self.max_loss):
#                     self.loss_queue.put((loss_mass,ms2[1],ms2[2],ms2[3],ms2[4],ms2[5]))


#     def make_corpus_from_queue(self):
#         self.corpus = {}
#         for file_name in self.files:
#             self.corpus[file_name] = {}
#         self.n_words = 0
#         self.word_counts = {}
#         self.seed_counts = {}
#         self.process_queue(self.fragment_queue,self.fragment_tol,'fragment')
#         self.process_queue(self.loss_queue,self.loss_tol,'loss')
#         print "\t Found {} words after grouping".format(self.n_words)



#     def get_first_file(self):
#         file_name = self.files[0]
#         return self.corpus[file_name]

#     def list_words(self,word_type,min_mass = 0.0,max_mass = 1000.0):
#         for word in self.word_counts:
#             if word.startswith(word_type):
#                 word_mass = float(word.split('_')[1])
#                 if (word_mass > min_mass and word_mass < max_mass):
#                     print "{}: {}".format(word,self.word_counts[word])

#     def remove_zero_words(self):
#         to_remove = []
#         for word in self.word_counts:
#             if self.word_counts[word] == 0:
#                 to_remove.append(word)
#         for word in to_remove:
#             del self.word_counts[word]
#             if word in self.seed_counts:
#                 del self.seed_counts[word]
#         print "\t Removed {} words".format(len(to_remove))
#         print "\t Finished, {} words (including {} seeds)".format(len(self.word_counts),len(self.seed_counts))

#     def process_queue(self,q,tolerance,prefix):
#         current_item = q.get()
#         current_mass = current_item[0]
#         sub_list = [current_item]
#         while not q.empty():
#             new_item = q.get()
#             new_mass = new_item[0]
#             if 1e6*abs((new_mass - current_mass)/new_mass) < tolerance:
#                 sub_list.append(new_item)
#             else:
#                 # this is a new group
#                 tot_mass = 0.0
#                 seeds = []
#                 for item in sub_list:
#                     if item[-1] == None:
#                         seeds.append(item)
#                     tot_mass += item[0]

#                 if len(seeds) == 0:
#                     mean_mass = tot_mass / (1.0*len(sub_list))
#                     word_name = prefix + "_{}".format(mean_mass)
#                     # self.word_counts[word_name] = len(sub_list)
#                     self.n_words += 1
#                 else:
#                     seed_names = []
#                     seed_masses = []
#                     for seed in seeds:
#                         seed_names.append(prefix + "_{}".format(seed[0]))
#                         seed_masses.append(seed[0])
#                         self.word_counts[seed_names[-1]] = 0
#                         self.seed_counts[seed_names[-1]] = 0

#                     self.n_words += len(seeds)
#                     seed_masses = np.array(seed_masses)
#                 for item in sub_list:
#                     if not item[-1] == None:
#                         # It's not a seed
#                         file_name = item[4]
#                         doc_name = item[3].name
#                         if not doc_name in self.corpus[file_name]:
#                             self.corpus[file_name][doc_name] = {}

#                         if len(seeds) > 0:
#                             # Find the closest seed
#                             di = np.abs(item[0] - seed_masses)
#                             word_name = seed_names[di.argmin()]
#                             self.word_counts[word_name] += 1
#                             self.seed_counts[word_name] += 1

#                         if word_name in self.corpus[file_name][doc_name]:
#                             self.corpus[file_name][doc_name][word_name] += item[2]
#                         else:
#                             if not word_name in self.word_counts:
#                                 self.word_counts[word_name] = 0
#                             self.word_counts[word_name] += 1
#                             self.corpus[file_name][doc_name][word_name] = item[2]


#                 sub_list = [new_item]
#                 current_mass = new_item[0]
#         if len(sub_list) > 0:
#                 # this is a new group
#             tot_mass = 0.0
#             seeds = []
#             for item in sub_list:
#                 if item[-1] == None:
#                     seeds.append(item)
#                 tot_mass += item[0]

#             if len(seeds) == 0:
#                 mean_mass = tot_mass / (1.0*len(sub_list))
#                 word_name = prefix + "_{}".format(mean_mass)
#                 self.word_counts[word_name] = len(sub_list)
#                 self.n_words += 1
#             else:
#                 seed_names = []
#                 seed_masses = []
#                 for seed in seeds:
#                     seed_names.append(prefix + "_{}".format(seed[0]))
#                     seed_masses.append(seed[0])
#                     self.word_counts[seed_names[-1]] = 0
                    
#                 self.n_words += len(seeds)
#                 seed_masses = np.array(seed_masses)
#             for item in sub_list:
#                 if not item[-1] == None:
#                     # It's not a seed
#                     file_name = item[4]
#                     doc_name = item[3].name
#                     if not doc_name in self.corpus[file_name]:
#                         self.corpus[file_name][doc_name] = {}

#                     if len(seeds) > 0:
#                         # Find the closest seed
#                         di = np.abs(item[0] - seed_masses)
#                         word_name = seed_names[di.argmin()]
#                         self.word_counts[word_name] += 1

#                     if word_name in self.corpus[file_name][doc_name]:
#                         self.corpus[file_name][doc_name][word_name] += item[2]
#                     else:
#                         self.corpus[file_name][doc_name][word_name] = item[2]

#     def load_gnps(self,merge_energies = True,merge_ppm = 2):
#         self.files = ['gnps']
#         self.ms1 = []
#         self.ms1_index = {}
#         self.ms2 = []
#         self.metadata = {}
#         n_processed = 0
#         ms2_id = 0
#         self.metadata = {}
#         for file in self.input_set:
#             with open(file,'r') as f:
#                 temp_mass = []
#                 temp_intensity = []
#                 doc_name = file.split('/')[-1]
#                 self.metadata[doc_name] = {}
#                 new_ms1 = MS1(str(n_processed),None,None,None,'gnps')
#                 new_ms1.name = doc_name
#                 self.ms1.append(new_ms1)
#                 self.ms1_index[str(n_processed)] = new_ms1
#                 for line in f:
#                     rline = line.rstrip()
#                     if len(rline) > 0:
#                         if rline.startswith('>'):
#                             keyval = rline[1:].split(' ')[0]
#                             valval = rline[len(keyval)+2:]
#                             if not keyval == 'ms2peaks':
#                                 self.metadata[doc_name][keyval] = valval
#                             if keyval == 'compound':
#                                 self.metadata[doc_name]['annotation'] = valval
#                             if keyval == 'parentmass':
#                                 self.ms1[-1].mz = float(valval)
#                             if keyval == 'intensity':
#                                 self.ms1[-1].intensity = float(valval)
#                         else:
#                             # If it gets here, its a fragment peak
#                             sr = rline.split(' ')
#                             mass = float(sr[0])
#                             intensity = float(sr[1])
#                             if intensity > self.min_intensity:
#                                 if merge_energies and len(temp_mass)>0:
#                                     errs = 1e6*np.abs(mass-np.array(temp_mass))/mass
#                                     if errs.min() < merge_ppm:
#                                         # Don't add, but merge the intensity
#                                         min_pos = errs.argmin()
#                                         if self.replace == 'max':
#                                             temp_intensity[min_pos] = max(intensity,temp_intensity[min_pos])
#                                         else:
#                                             temp_intensity[min_pos] += intensity
#                                     else:
#                                         temp_mass.append(mass)
#                                         temp_intensity.append(intensity)
#                                 else:
#                                     temp_mass.append(mass)
#                                     temp_intensity.append(intensity)

#                 parent = self.ms1[-1]
#                 for mass,intensity in zip(temp_mass,temp_intensity):
#                     new_ms2 = (mass,0.0,intensity,parent,'gnps',float(ms2_id))
#                     self.ms2.append(new_ms2)
#                     ms2_id += 1

#                 n_processed += 1
#             if n_processed % 100 == 0:
#                 print "Processed {} spectra".format(n_processed)
            
#     def load_metfamily_matrix(self):
#         self.files = ['metfamily']
#         self.corpus = {}
#         self.corpus['metfamily'] = {}
#         self.metadata = {}
#         self.word_counts = {}
#         with open(self.input_set,'r') as f:
#             line = f.readline()
#             line = f.readline()
#             # headings line
#             line = f.readline()
#             heads = line.rstrip().split('\t')
#             sample_names = heads[17:23]
#             raw_feature_names = heads[23:]
#             feature_names = []
#             for feat in raw_feature_names:
#                 if feat.startswith('-'):
#                     feature_names.append('loss_{}'.format(feat[1:]))
#                 else:
#                     feature_names.append('fragment_{}'.format(feat))

#             for feat in feature_names:
#                 self.word_counts[feat] = 0

#             print "Samples:", sample_names
#             max_metadata_pos = 16

#             n_docs = 0
#             for line in f:
#                 tokens = line.rstrip('\n').split('\t')
#                 mz = tokens[0]
#                 rt = tokens[1]
#                 doc_name = "{}_{}".format(mz,rt)
#                 self.metadata[doc_name] = {}
#                 self.corpus['metfamily'][doc_name] = {}
#                 for i in range(max_metadata_pos + 1):
#                     key = heads[i]
#                     value = tokens[i]
#                     self.metadata[doc_name][key] = value
#                 self.metadata[doc_name]['intensities'] = dict(zip(sample_names,[float(i) for i in tokens[17:23]]))
#                 feats = [(index,intensity) for index,intensity in enumerate(tokens[23:]) if len(intensity) > 0]
#                 for index,intensity in feats:
#                     self.corpus['metfamily'][doc_name][feature_names[index]] = float(intensity)
#                     self.word_counts[feature_names[index]] += 1
                
#                 n_docs += 1
#                 if n_docs % 100 == 0:
#                     print n_docs

#         print "Loaded {} documents, and {} unique words ({} total word instances)".format(n_docs,
#                                                                                           len(self.word_counts),
#                                                                                           sum(self.word_counts.values()))


#     def load_csv(self):
#         # input set is a list of tuples, each one has ms1 and ms2 files
#         # Load the ms1 files
#         self.files = []
#         self.ms1 = []
#         self.ms1_index = {}
#         self.ms2 = []
#         for input in self.input_set:
#             file_name = input[0].split('/')[-1].split('_ms1')[0]
#             print file_name
#             self.files.append(file_name)
#             with open(input[0],'r') as f:
#                 heads = f.readline().split(',') # remove headings
#                 rt_pos = heads.index('"rt"')
#                 mz_pos = heads.index('"mz"')
#                 intensity_pos = heads.index('"intensity"')
#                 peak_id_pos = heads.index('"peakID"')
#                 for line in f:
#                     tokens = line.split(',')
#                     rt = float(tokens[rt_pos])
#                     mz = float(tokens[mz_pos])
#                     intensity = float(tokens[intensity_pos])
#                     id = tokens[peak_id_pos]
#                     new_ms1 = MS1(id,mz,rt,intensity,file_name)
#                     self.ms1.append(new_ms1)
#                     self.ms1_index[id] = new_ms1
#             print "\t loaded {} ms1 peaks".format(len(self.ms1))

#             with open(input[1],'r') as f:
#                 heads = f.readline().split(',')
#                 rt_pos = heads.index('"rt"')
#                 mz_pos = heads.index('"mz"')
#                 intensity_pos = heads.index('"intensity"')
#                 peak_id_pos = heads.index('"peakID"')
#                 parent_id_pos = heads.index('"MSnParentPeakID"')
#                 for line in f:
#                     tokens = line.split(',')
#                     rt = float(tokens[rt_pos])
#                     mz = float(tokens[mz_pos])
#                     intensity = float(tokens[intensity_pos])                    
#                     id = tokens[peak_id_pos]
#                     parent_id = tokens[parent_id_pos]
#                     parent = self.ms1_index[parent_id]
#                     self.ms2.append((mz,rt,intensity,parent,file_name,id))
                
#                 print "\t loaded {} ms2 peaks".format(len(self.ms2))

                    


