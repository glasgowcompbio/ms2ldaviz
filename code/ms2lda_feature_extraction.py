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
    def __init__(self,id,mz,rt,intensity,file_name,scan_number = None):
        self.id = id
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.file_name = file_name
        self.scan_number = scan_number
        self.name = "{}_{}".format(self.mz,self.rt)

    def __str__(self):
        return self.name






# ******************************
# ******************************
# ******************************
# LOADERS
# ******************************
# ******************************
# ******************************




# Abstract loader class
class Loader(object):
    def load_spectra(self,input_set):
        raise NotImplementedError("load spectra method must be implemented")


# A class to load the data from the metfamily paper
class LoadMetfamily(Loader):
    def __str__(self):
        return "Object to load metfamily formated fragment matrix"
    def load_spectra(self,input_set):
        filename = 'metfamily'
        self.corpus = {}
        self.corpus[filename] = {}
        self.metadata = {}
        self.word_counts = {}
        self.word_mz_range = {}
        self.ms1 = []
        ms1_id = 0
        if type(input_set) == list:
            input_set = input_set[0]
        with open(input_set,'r') as f:
            # chop empty lines
            line = f.readline()
            line = f.readline()
            # headings line
            line = f.readline()
            heads = line.rstrip().split('\t')
            # Grab the sample names
            sample_names = heads[17:23]
            raw_feature_names = heads[23:]

            # Convert the feature names into our format
            feature_names = []
            for feat in raw_feature_names:
                if feat.startswith('-'):
                    feature_names.append('loss_{}'.format(feat[1:]))
                else:
                    feature_names.append('fragment_{}'.format(feat))

            # Create some useful objects.
            # Note that the word range is None,None as we cant reverse engineer this
            for feat in feature_names:
                self.word_counts[feat] = 0
                self.word_mz_range[feat] = (None,None)

            max_metadata_pos = 16 # This should be checked if ever loading a different file.


            # The major loading bit
            for line in f:
                tokens = line.rstrip('\n').split('\t')
                # Grab the document details
                mz = tokens[0]
                rt = tokens[1]
                new_ms1 = MS1(ms1_id,mz,rt,None,filename)
                ms1_id += 1
                self.ms1.append(new_ms1)
                doc_name = new_ms1.name

                self.metadata[doc_name] = {}
                self.corpus[filename][doc_name] = {}
                for i in range(max_metadata_pos + 1):
                    key = heads[i]
                    value = tokens[i]
                    self.metadata[doc_name][key] = value

                # Add the intensities
                self.metadata[doc_name]['intensities'] = dict(zip(sample_names,[float(i) for i in tokens[17:23]]))
                feats = [(index,intensity) for index,intensity in enumerate(tokens[23:]) if len(intensity) > 0]
                for index,intensity in feats:
                    self.corpus['metfamily'][doc_name][feature_names[index]] = float(intensity)
                    self.word_counts[feature_names[index]] += 1

        # Return the corpus and word range in place of the ms2 list
        return self.ms1,(self.corpus,self.word_mz_range),self.metadata




# A class to load files in Joe's csv format
# This is yet to be tested
class LoadCSV(Loader):

    def load_spectra(self,input_set):
        # input set is a list of tuples, each one has ms1 and ms2 files
        # Load the ms1 files
        self.files = []
        self.ms1 = []
        self.ms1_index = {}
        self.ms2 = []
        self.metadata = {}
        self.input_set = input_set
        for input in self.input_set:
            file_name = input[0].split('/')[-1].split('_ms1')[0]
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
                    doc_name = new_ms1.name
                    self.metadata[doc_name] = {}
                    self.metadata[doc_name]['parentmass'] = mz
                    self.metadata[doc_name]['parentrt'] = rt
                    self.metadata[doc_name]['parentintensity'] = intensity
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
        return self.ms1,self.ms2,self.metadata


# A class to load mzml files
# Will ultimately be able to do method 3

# This method finds each ms2 spectrum in the file and then looks back at the last ms1 scan to find the most
# intense ms1 peak within plus and minus the isolation window. If nothing is found, no document is created
# If it is found, a document is created

# If a peak list is provided it then tries to match the peaks in the peaklist to the ms1 objects, just
# keeping the ms1 objects that can be matched. The matching is done with plus and minus the mz_tol (ppm)
# and plus and minus the rt_tol
class LoadMZML(Loader):
    def __init__(self,min_intensity = 0.0,peaklist = None,isolation_window = 1.0,mz_tol = 5,rt_tol=5.0):
        self.min_intensity = min_intensity
        self.peaklist = peaklist
        self.isolation_window = isolation_window
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
    def __str__(self):
        return "mzML loader"
    def load_spectra(self,input_set):
        import pymzml
        import bisect

        ms1 = []
        ms2 = []
        metadata = {}
        nc = 0
        ms2_id = 0
        ms1_id = 0


        for input_file in input_set:
            print "Loading spectra from {}".format(input_file)
            current_ms1_scan_mz = None
            current_ms1_scan_intensity = None
            current_ms1_scan_rt = None
            run = pymzml.run.Reader(input_file, MS1_Precision=5e-6)
            file_name = input_file.split('/')[-1]
            for spectrum in run:
                if spectrum['ms level'] == 1:
                    current_ms1_scan_rt = spectrum['scan start time']
                    current_ms1_scan_mz,current_ms1_scan_intensity = zip(*spectrum.peaks)
                elif spectrum['ms level'] == 2:
                    precursor_mz = spectrum['precursors'][0]['mz']
                    # This finds the insertion position for the precursor mz (i.e. the position one to the right
                    # of the first element it is greater than)
                    precursor_index_ish = bisect.bisect_right(current_ms1_scan_mz,precursor_mz)
                    pos = precursor_index_ish - 1 # pos is now the largest value smaller than ours

                    # Move left and right within the precursor window and pick the most intense parent_scan_number
                    max_intensity = 0.0
                    max_intensity_pos = None
                    while abs(precursor_mz - current_ms1_scan_mz[pos]) < self.isolation_window:
                        if current_ms1_scan_intensity[pos] >= max_intensity:
                            max_intensity = current_ms1_scan_intensity[pos]
                            max_intensity_pos = pos
                        pos -= 1
                        if pos < 0:
                            break
                    pos = precursor_index_ish
                    if pos < len(current_ms1_scan_mz):
                        while abs(precursor_mz - current_ms1_scan_mz[pos]) < self.isolation_window:
                            if current_ms1_scan_intensity[pos] >= max_intensity:
                                max_intensity = current_ms1_scan_intensity[pos]
                                max_intensity_pos = pos
                            pos += 1
                            if pos > len(current_ms1_scan_mz)-1:
                                break
                        # print current_ms1_scan_mz[max_intensity_pos],current_ms1_scan_rt
                    # Make the new MS1 object
                    if not max_intensity_pos == None:
                    # mz,rt,intensity,file_name,scan_number = None):
                        new_ms1 = MS1(ms1_id,current_ms1_scan_mz[max_intensity_pos],
                                      current_ms1_scan_rt,max_intensity,file_name,scan_number = nc)
                        metadata[new_ms1.name] = {'parentmass':current_ms1_scan_mz[max_intensity_pos],
                                                  'parentrt':current_ms1_scan_rt,'scan_number':nc,
                                                  'precursor_mass':precursor_mz}

                        ms1.append(new_ms1)
                        ms1_id += 1

                        # Make the ms2 objects:
                        for mz,intensity in spectrum.centroidedPeaks:
                            ms2.append((mz,current_ms1_scan_rt,intensity,new_ms1,file_name,float(ms2_id)))
                            ms2_id += 1

        print "Found {} ms2 spectra, and {} individual ms2 objects".format(len(ms1),len(ms2))


        if self.peaklist:
            ms1_peaks = self._load_peak_list()
            ms1 = sorted(ms1,key = lambda x: x.mz)
            new_ms1_list = []
            new_ms2_list = []
            new_metadata = {}
            # ms1_mz = [x.mz for z in ms1]
            n_peaks_checked = 0
            for n_peaks_checked,peak in enumerate(self.ms1_peaks):
                if n_peaks_checked % 500 == 0:
                    print n_peaks_checked
                peak_mz = peak[0]
                peak_rt = peak[1]
                peak_intensity = peak[2]

                min_mz = peak_mz - self.mz_tol*peak_mz/1e6
                max_mz = peak_mz + self.mz_tol*peak_mz/1e6
                min_rt = peak_rt - self.rt_tol
                max_rt = peak_rt + self.rt_tol


                ms1_hits = filter(lambda x: x.mz >= min_mz and x.mz <= max_mz and x.rt >= min_rt and x.rt <= max_rt,ms1)

                if len(ms1_hits) == 1:
                    # Found one hit, easy
                    old_ms1 = ms1_hits[0]
                elif len(ms1_hits) > 1:
                    # Find the one with the most intense MS2 peak
                    best_ms1 = None
                    best_intensity = 0.0
                    for frag_peak in ms2:
                        if frag_peak[3] in ms1_hits:
                            if frag_peak[2] > best_intensity:
                                best_intensity = frag_peak[2]
                                best_ms1 = frag_peak[3]
                    old_ms1 = best_ms1
                else:
                    # Didn't find any
                    continue

                # make a new ms1 object
                new_ms1 = MS1(old_ms1.id,peak_mz,peak_rt,peak_intensity,old_ms1.file_name,old_ms1.scan_number)
                new_ms1_list.append(new_ms1)
                new_metadata[new_ms1.name] = metadata[old_ms1.name]

                # Delete the old one so it can't be picked again - removed this, maybe it's not a good idea?
                # pos = ms1.index(old_ms1)
                # del ms1[pos]

                # Change the reference in the ms2 objects to the new ms1 object
                ms2_objects = filter(lambda x: x[3] == old_ms1,ms2)
                for frag_peak in ms2_objects:
                    new_frag_peak = (frag_peak[0],peak_rt,frag_peak[2],new_ms1,frag_peak[4],frag_peak[5])
                    new_ms2_list.append(new_frag_peak)

            # replace the ms1,ms2 and metadata with the new versions
            ms1 = new_ms1_list
            ms2 = new_ms2_list
            metadata = new_metadata
            print "Peaklist filtering results in {} documents".format(len(ms1))

        return ms1,ms2,metadata

    def _load_peak_list(self):
        self.ms1_peaks = []
        with open(self.peaklist,'r') as f:
            heads = f.readline()
            for line in f:
                tokens = line.split(',')
                # get mx,rt,intensity
                self.ms1_peaks.append((float(tokens[1]),float(tokens[2]),float(tokens[3])))

        # sort them by mass
        self.ms1_peaks = sorted(self.ms1_peaks,key = lambda x: x[0])
        print "Loaded {} ms1 peaks from {}".format(len(self.ms1_peaks),self.peaklist)


# A class to load Emma's data
# This is very rough
# pass cid as spectrum_choice for collision-induced dissociation
# pass hcid for beam-type collision-induced dissociation
class LoadEmma(Loader):
    def __init__(self,min_intensity = 0.0,spectrum_choice = 'cid',peaklist = None,mz_tol = 10,rt_tol = 0.5):
        self.spectrum_choice = spectrum_choice # choices are cid and hcid
        self.min_intensity = min_intensity
        self.peaklist = peaklist
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
    def __str__(self):
        return "Loader for Emma data format"
    def load_spectra(self,input_set):
        import pymzml

        ms1 = []
        ms2 = []
        metadata = {}
        nc = 0
        ms2_id = 0


        if self.peaklist:
            ms1_peak_list = self._load_peak_list()
            peak_list_pos = 0
        else:
            ms1_peak_list = None



        for input_file in input_set:

            # Load the mzml object
            run = pymzml.run.Reader(input_file, MS1_Precision=5e-6)
            nspec = 0
            nmatch = 0
            parent_scan_number = None
            file_name = input_file.split('/')[-1]

            for spectrum in run:
                if 'collision-induced dissociation' in spectrum:
                    if self.spectrum_choice == 'cid':
                        parentmass = spectrum['precursors'][0]['mz']
                        parentrt = spectrum['scan start time']
                        nc += 1
                        newms1 = MS1(nc,parentmass,parentrt,None,file_name,scan_number = parent_scan_number)
                        ms1.append(newms1)
                        metadata[newms1.name] = {'parentmass':parentmass,'parentrt':parentrt}
                        for mz,intensity in spectrum.peaks:
                            if intensity > self.min_intensity:
                                ms2.append((mz,parentrt,intensity,ms1[-1],file_name,float(ms2_id)))
                                ms2_id += 1
                elif 'beam-type collision-induced dissociation' in spectrum:
                    if self.spectrum_choice == 'hcid':
                        parentmass = spectrum['precursors'][0]['mz']
                        parentrt = spectrum['scan start time']
                        nc += 1
                        newms1 = MS1(nc,parentmass,parentrt,None,file_name,scan_number = parent_scan_number)
                        ms1.append(newms1)
                        metadata[newms1.name] = {'parentmass':parentmass,'parentrt':parentrt}
                        for mz,intensity in spectrum.peaks:
                            if intensity > self.min_intensity:
                                ms2.append((mz,parentrt,intensity,ms1[-1],file_name,float(ms2_id)))
                                ms2_id += 1
                else:
                    # This is an MS1 scan update our parent scan number
                    parent_scan_number = nspec + 1
                nspec += 1


        if ms1_peak_list:
            print "Filtering based on peak list"
            filtered_ms1_list = []
            # A peak list has been loaded so we need to filter the MS1 peaks
            ms1 = sorted(ms1,key = lambda x: x.mz)
            for i,peak in enumerate(ms1_peak_list):
                mz = peak[1]
                rt = peak[2]
                pos = 0
                hits = []
                for ms in ms1:
                    m = self._peak_mass_match(mz,ms.mz)
                    if m == True:
                        if self._peak_rt_match(rt,ms.rt):
                            hits.append((ms,abs(mz-ms.mz)/mz))
                    else:
                        if 1e6*(ms.mz-mz)/mz > self.mz_tol:
                            # Gone too far, break from the loop
                            break
                if len(hits) > 0:
                    hits = sorted(hits,key = lambda x: x[1])
                    top_hit = hits[0][0]
                    filtered_ms1_list.append(top_hit)
            print "Filtering results in {} hits".format(len(filtered_ms1_list))

            filtered_ms2_list = []
            for m in ms2:
                if m[3] in filtered_ms1_list:
                    filtered_ms2_list.append(m)

            print "Filtering results in {} ms2 peaks".format(len(filtered_ms2_list))

            ms1 = filtered_ms1_list
            ms2 = filtered_ms2_list


        return ms1,ms2,metadata


    def _peak_mass_match(self,mz1,mz2):
        if 1e6*abs(mz1-mz2)/mz1 < self.mz_tol:
            return True
        else:
            return False



    def _peak_rt_match(self,rt1,rt2):
        if abs(rt1-rt2) < self.rt_tol:
            return True
        else:
            return False

    def _load_peak_list(self):
        ms1_peak_list = []
        with open(self.peaklist,'r') as f:
            heads = f.readline().split('\t')
            try:
                mz_pos = heads.index('Centroid m/z')
                rt_pos = heads.index('RT (min.)')
                scan_pos = heads.index('Scan Number')
                start_scan = heads.index('Start Scan Number')
                end_scan = heads.index('End Scan Number')
                for line in f:
                    tokens = line.split()
                    ms1_peak_list.append((int(tokens[scan_pos]),float(tokens[mz_pos]),float(tokens[rt_pos]),int(tokens[start_scan]),int(tokens[end_scan])))
                ms1_peak_list = sorted(ms1_peak_list,key = lambda x: x[1]) # Sort by mass
                print "Loaded {} peaks from peak list".format(len(ms1_peak_list))
                return ms1_peak_list
            except:
                print "Can't load peaklist, using none"
                return None

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
                            if len(tokens) == 2:
                                # One tuple per line
                                mz = float(tokens[0])
                                intensity = float(tokens[1])
                                ms2.append((mz,0.0,intensity,new_ms1,file_name,float(ms2_id)))
                                ms2_id += 1
                            else:
                                for pos in range(0,len(tokens),2):
                                    mz = float(tokens[pos])
                                    intensity = float(tokens[pos+1])
                                    ms2.append((mz,0.0,intensity,new_ms1,file_name,float(ms2_id)))
                                    ms2_id += 1

                        elif rline.startswith('Num Peaks'):
                            in_doc = True
                            new_ms1 = MS1(ms1_id,parentmass,parentrt,None,file_name)
                            ms1_id += 1
                            if 'Name' in temp_metadata:
                                doc_name = temp_metadata['Name']
                            else:
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













# ******************************
# ******************************
# ******************************
# Feature extractors and corpus makers
# ******************************
# ******************************
# ******************************



# Abstract feature making class
class MakeFeatures(object):
    def make_features(self,ms2):
        raise NotImplementedError("make features method must be implemented")

# Class to make non-processed features
# i.e. no feature matching, just the raw value
class MakeRawFeatures(MakeFeatures):
    def __str__(self):
        return 'raw feature maker'
    def __init__(self):
        pass
    def make_features(self,ms2):
        self.word_mz_range = {}
        self.corpus = {}

        for peak in ms2:
            frag_mz = peak[0]
            frag_intensity = peak[2]
            parent = peak[3]
            doc_name = parent.name
            file_name = peak[4]

            if not file_name in self.corpus:
                self.corpus[file_name] = {}
            if not doc_name in self.corpus[file_name]:
                self.corpus[file_name][doc_name] = {}

            feature_name = 'fragment_{}'.format(frag_mz)
            self.corpus[file_name][doc_name][feature_name] = frag_intensity
            self.word_mz_range[feature_name] = (frag_mz,frag_mz)

        return self.corpus,self.word_mz_range



# Class to make features by binning with width bin_width
class MakeBinnedFeatures(MakeFeatures):

    def __str__(self):
        return "Binning feature creator with bin_width = {}".format(self.bin_width)

    def __init__(self,min_frag = 0.0,max_frag = 1000.0,
                          min_loss = 10.0,max_loss = 200.0,
                          min_intensity = 0.0,min_intensity_perc = 0.0,bin_width = 0.005):
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity
        self.bin_width = bin_width
        self.min_intensity_perc = min_intensity_perc

    def make_features(self,ms2):
        self.word_mz_range = {}
        self.word_counts = {}
        self.fragment_words = []
        self.loss_words = []
        self.corpus = {}

        self._make_words(self.min_loss,self.max_loss,self.loss_words,'loss')
        self._make_words(self.min_frag,self.max_frag,self.fragment_words,'fragment')

        for word in self.fragment_words:
            self.word_counts[word] = 0
        for word in self.loss_words:
            self.word_counts[word] = 0


        # make a list of the lower edges
        frag_lower = [self.word_mz_range[word][0] for word in self.fragment_words]
        loss_lower = [self.word_mz_range[word][0] for word in self.loss_words]

        import bisect


        for peak in ms2:

            # MS2 objects are ((mz,rt,intensity,parent,file_name,id))
            # TODO: make the search more efficients
            mz = peak[0]
            if peak[3].mz == None:
                # There isnt a precursor mz so we cant do losses
                do_losses = False
                loss_mz = 0.0
            else:
                do_losses = True
                loss_mz = peak[3].mz - mz
            intensity = peak[2]
            if intensity >= self.min_intensity:
                doc_name = peak[3].name
                file_name = peak[4]
                if mz > self.min_frag and mz < self.max_frag:
                    pos = bisect.bisect_right(frag_lower,mz)
                    word = self.fragment_words[pos-1]
                    if not file_name in self.corpus:
                        self.corpus[file_name] = {}
                    if not doc_name in self.corpus[file_name]:
                        self.corpus[file_name][doc_name] = {}
                    if not word in self.corpus[file_name][doc_name]:
                        self.corpus[file_name][doc_name][word] = 0.0
                        self.word_counts[word] += 1
                    self.corpus[file_name][doc_name][word] += intensity

                if do_losses and loss_mz > self.min_loss and loss_mz < self.max_loss:
                    pos = bisect.bisect_right(loss_lower,loss_mz)
                    word = self.loss_words[pos-1]
                    if not file_name in self.corpus:
                        self.corpus[file_name] = {}
                    if not doc_name in self.corpus[file_name]:
                        self.corpus[file_name][doc_name] = {}
                    if not word in self.corpus[file_name][doc_name]:
                        self.corpus[file_name][doc_name][word] = 0.0
                        self.word_counts[word] += 1
                    self.corpus[file_name][doc_name][word] += intensity
                    

        # TODO: Test code to remove blank words!!!!!
        to_remove = []
        for word in self.word_mz_range:
            if self.word_counts[word] == 0:
                to_remove.append(word)

        for word in to_remove:
            del self.word_mz_range[word]

        n_docs = 0
        for c in self.corpus:
            n_docs += len(self.corpus[c])

        print "{} documents".format(n_docs)
        print "After removing empty words, {} words left".format(len(self.word_mz_range))

        if self.min_intensity_perc > 0:
            # Remove words that are smaller than a certain percentage of the highest feature
            for c in self.corpus:
                for doc in self.corpus[c]:
                    max_intensity = 0.0
                    for word in self.corpus[c][doc]:
                        intensity = self.corpus[c][doc][word]
                        if intensity > max_intensity:
                            max_intensity = intensity
                    to_remove = []
                    for word in self.corpus[c][doc]:
                        intensity = self.corpus[c][doc][word]
                        if intensity < max_intensity * self.min_intensity_perc:
                            to_remove.append(word)
                    for word in to_remove:
                        del self.corpus[c][doc][word]
                        self.word_counts[word] -= 1
            # Remove any words that are now empty
            to_remove = []
            for word in self.word_mz_range:
                if self.word_counts[word] == 0:
                    to_remove.append(word)
            for word in to_remove:
                del self.word_mz_range[word]
            print "After applying min_intensity_perc filter, {} words left".format(len(self.word_mz_range))


        return self.corpus,self.word_mz_range

    def _make_words(self,min_mz,max_mz,word_list,prefix):
        # Bin the ranges to make the words
        min_word = float(min_mz)
        max_word = min_word + self.bin_width
        while min_word < max_mz:
            up_edge = min(max_mz,max_word)
            word_mean = 0.5*(min_word + up_edge)
            new_word = '{}_{}'.format(prefix,word_mean)
            word_list.append(new_word)
            self.word_mz_range[new_word] = (min_word,up_edge)
            min_word += self.bin_width
            max_word += self.bin_width






# Class to make the metfamily features.
# This class does almost nnothing as the corpus is already defined in the data matrix.
# It just needs to split the 'ms2' object returned by the loader into its two parts: the corpus and the word ranges

class MakeMetfamilyFeatures(MakeFeatures):
    def __str__(self):
        return "Dummy classs to make metfamily corpus, although its already made by the loader"
    def make_features(self,ms2):
        # Return the corpus and the word_mz_range objects
        self.corpus = ms2[0]
        self.word_mz_range = ms2[1]
        return self.corpus,self.word_mz_range


# Class to make nominal (i.e. integer features)
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
        return self.corpus,self.word_mz_range



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

    def get_first_corpus(self):
        first_file_name = self.corpus.keys()[0]
        return self.corpus[first_file_name]




