from pprint import pprint
import glob
import copy
import gzip
import cPickle
from operator import itemgetter

import numpy as np
import pandas as pd
import networkx as nx

from gnps_feature import FeatureExtractor

class Peak:

    def __init__(self, peak_id, mz, rt, intensity):
        self.peak_id = peak_id
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.level = 2

    def get_closest(self, others, nnz):
        other_mzs = np.array([x.mz for x in others])
        diff = np.abs(self.mz - other_mzs)
        min_idx = np.argmin(diff)
        return others[min_idx], nnz[min_idx]

    def __str__(self):
      return 'mz=%.5f rt=%.3f intensity=%f' % (self.mz, self.rt, self.intensity)

    def __repr__(self):
      return self.__str__()

class Document:

    def __init__(self, data):

        self.doc_id = data['doc_id']
        self.compound = data['compound']
        self.formula = data['formula']
        self.parentmass = float(data['parentmass'])
        self.ionisation = data['ionization']
        self.inchi = data['InChI']
        self.inchikey = data['InChIKey']
        self.smiles = data['smiles']

        # self.spectra is a list. If its length is 3, then it's from 3 energy levels
        self.spectra = data['ms2peaks']
        self.words = []
        self.merged = False

    # merge spectra from different energy levels into one spectrum
    def merge_spectra(self, mass_tol):

        if self.merged: # has been merged before
            return

        # if only one spectrum, then flatten this list
        if len(self.spectra) == 1:
            self.spectra = self.spectra[0]
            self.merged = True
            return

        # else we need to merge all the spectra
        spectra = copy.deepcopy(self.spectra)

        # set the largest spectrum as the reference
        last_spectra = len(spectra)-1
        merged = spectra[last_spectra]

        # successively merge each spectrum towards the reference
        for i in range(last_spectra, 0, -1):

            to_process = spectra[i-1]
            for peak in to_process:

                others = np.array([x.mz for x in merged])
                matches = self._mass_match(peak.mz, others, mass_tol)
                nnz = np.nonzero(matches)[0]

                if len(nnz) > 1: # multiple matches

                    # take the closest in mass
                    candidates = [merged[idx] for idx in nnz]
                    closest, pos = peak.get_closest(candidates, nnz)
                    if closest.intensity < peak.intensity:
                        merged[pos].intensity = peak.intensity

                elif len(nnz) == 0: # no matches
                    merged.append(peak)

                else: # exactly one match
                    pos = nnz[0]
                    if merged[pos].intensity < peak.intensity:
                        merged[pos].intensity = peak.intensity

        # sort the list in place by m/z
        merged.sort(key=lambda x: x.mz, reverse=False)
        self.spectra = merged
        self.merged = True

    def _mass_match(self, mass, other_masses, tol):
        return np.abs((mass-other_masses)/mass)<tol*1e-6

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as f:
        obj = cPickle.load(f)
        return obj

def save_object(obj, filename):
    with gzip.GzipFile(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_gnps_data(pattern):

    collection = [] # to store the entire results
    filenames = glob.glob(pattern)
    peak_id = 0

    print 'Total files = %d' % len(filenames)
    for i in range(len(filenames)):

        if i > 0 and i % 1000 == 0:
            print 'Loading %d/%d' % (i, len(filenames))

        fn = filenames[i]
        with open(fn) as f:

            # read all the lines, remove \n at the end of each line
            content = f.read().splitlines()

            data = {}
            spectra = [] # all the spectra in the document
            ms2 = [] # one MS2 spectrum

            for line in content:

                if not line: # if empty line, end of spectra reading state
                    if len(ms2)>0: # store an MS2 spectrum, if any
                        spectra.append(ms2)
                    continue

                if line.startswith('>'):

                    line = line[1:] # remove '>' from line
                    tokens = line.split(' ')
                    first = tokens[0]

                    # if the line starts with '>' but not '>ms2peaks'
                    if first != 'ms2peaks':
                        # then store that line content
                        line_content = ' '.join(tokens[1:])
                        data[first] = line_content # e.g. data['formula'] = 'C21H30O5'

                    # else if line is '>ms2peaks'
                    else:
                        # move to spectra reading state
                        ms2 = []

                else: # must be a line containing an MS2 peak

                    tokens = line.split(' ')
                    mz = float(tokens[0])
                    rt = 0
                    intensity = float(tokens[1])
                    peak = Peak(peak_id, mz, rt, intensity)
                    ms2.append(peak)
                    peak_id += 1

            if len(ms2)>0: # store the final MS2 spectrum, if any
                spectra.append(ms2)
            data['ms2peaks'] = spectra

            # initialise a document
            data['doc_id'] = 'doc_%d' % i
            doc = Document(data)
            collection.append(doc)

    return collection

def get_no_of_peaks(collection):
    no_peaks = 0
    for doc in collection:
        for spectrum in doc.spectra:
            no_peaks += len(spectrum)
    return no_peaks

def merge_spectra(collection, merge_tol):
    for i in range(len(collection)):
        if i > 0 and i % 1000 == 0:
            print 'Merging %d/%d' % (i, len(collection))
        doc = collection[i]
        doc.merge_spectra(merge_tol)

def normalise_spectra(collection):
    for doc in collection:
        max_intensity = float(max([peak.intensity for peak in doc.spectra]))
        for peak in doc.spectra:
            peak.intensity /= max_intensity

def filter_spectra(collection, min_intensity):
    for doc in collection:
        filtered = []
        for spectrum in doc.spectra:
            new_spectrum = [peak for peak in spectrum if peak.intensity > min_intensity]
            filtered.append(new_spectrum)
        doc.spectra = filtered

def get_dataframes(collection):

    ms1_peakids = []
    ms2_peakids = []
    ms1_peakdata = []
    ms2_peakdata = []

    peak_id = 0
    for doc in collection:

        ms1_id = int(peak_id)
        parent_id = np.nan
        ms_level = 1
        mz = doc.parentmass
        rt = 0
        intensity = 0

        peak_id += 1

        ms1_peakids.append(ms1_id)
        ms1_peakdata.append((ms1_id, parent_id, ms_level, mz, rt, intensity))

        for peak in doc.spectra:
            ms2_id = int(peak_id)
            parent_id = int(ms1_id)
            ms_level = 2
            mz = peak.mz
            rt = peak.rt
            intensity = peak.intensity
            ms2_peakdata.append((ms2_id, parent_id, ms_level, mz, rt, intensity))
            ms2_peakids.append(ms2_id)

            peak_id += 1

    ms1 = pd.DataFrame(ms1_peakdata, index=ms1_peakids,
                       columns=['peakID', 'MSnParentPeakID', 'msLevel', 'mz', 'rt', 'intensity'])
    ms2 = pd.DataFrame(ms2_peakdata, index=ms2_peakids,
                       columns=['peakID', 'MSnParentPeakID', 'msLevel', 'mz', 'rt', 'intensity'])

    return ms1, ms2

def get_corpus(collection, params):

    fragment_grouping_tol = params['fragment_grouping_tol']
    loss_grouping_tol = params['loss_grouping_tol']
    loss_threshold_min_val = params['loss_threshold_min_val']
    loss_threshold_max_val = params['loss_threshold_max_val']

    ms1, ms2 = get_dataframes(collection)
    input_set = [(ms1, ms2)]
    extractor = FeatureExtractor(input_set, fragment_grouping_tol, loss_grouping_tol,
                                          loss_threshold_min_val, loss_threshold_max_val)

    fragment_q = extractor.make_fragment_queue()
    fragment_groups = extractor.group_features(fragment_q, extractor.fragment_grouping_tol)
    loss_q = extractor.make_loss_queue()
    loss_groups = extractor.group_features(loss_q, extractor.loss_grouping_tol, check_threshold=True)

    extractor.create_counts(fragment_groups, loss_groups)
    lda_dict, vocab, ms1, ms2 = extractor.get_entry(0)
    return lda_dict, vocab, ms1, ms2

def filter_corpus(corpus_dict, min_count):

    # get total count
    total_counts = {}
    for doc_id in corpus_dict:
        doc = corpus_dict[doc_id]
        for feature in doc:
            count = doc[feature]
            if feature in total_counts:
                total_counts[feature] += count
            else:
                total_counts[feature] = count

    # create the set of features to keep
    keep = set()
    for feature, count in total_counts.items():
        if count > min_count:
            keep.add(feature)

    # do filtering
    filtered = {}
    thrown = 0
    kept = 0
    for doc_id in corpus_dict:

        doc = corpus_dict[doc_id]
        new_doc = {}

        for feature in doc:
            if feature in keep:
                count = doc[feature]
                new_doc[feature] = count
                kept += 1
            else:
                thrown += 1

        filtered[doc_id] = new_doc

    print kept, thrown
    return filtered