import sys

sys.path.append('../lda/code')
from ms2lda_feature_extraction import LoadMZML, MakeBinnedFeatures, LoadMSP, LoadMGF
from lda import VariationalLDA

from basicviz.models import Experiment, Document, Feature, FeatureInstance, Mass2Motif, Mass2MotifInstance, \
    DocumentMass2Motif, FeatureMass2MotifInstance, Alpha
import jsonpickle

# This is required for using motifdb motifs
class FeatureMatcher(object):
    def __init__(self,db_features,other_features,bin_width=0.005):
        self.db_features = db_features
        self.other_features = other_features
        self.fmap = {}
        self.bin_width = bin_width
        self.augmented_features = {f:v for f,v in other_features.items()}

        self.match()
        self.match(ftype='loss')



    def match(self,ftype='fragment'):
        import bisect
        other_names = [f for f in self.other_features if f.startswith(ftype)]
        other_min_mz = [self.other_features[f][0] for f in self.other_features if f.startswith(ftype)]
        other_max_mz = [self.other_features[f][1] for f in self.other_features if f.startswith(ftype)]

        temp = zip(other_names,other_min_mz,other_max_mz)
        temp.sort(key = lambda x: x[1])
        other_names,other_min_mz,other_max_mz = zip(*temp)
        other_names = list(other_names)
        other_min_mz = list(other_min_mz)
        other_max_mz = list(other_max_mz)

        exact_match = 0
        new_ones = 0
        overlap_match = 0
        for f in [f for f in self.db_features if f.startswith(ftype)]:
            if f in other_names:
                self.fmap[f] = f;
                exact_match += 1
            else:
                fmz = float(f.split('_')[1])
                if fmz < other_min_mz[0] or fmz > other_max_mz[-1]:
                    self.fmap[f] = f
                    self.augmented_features[f] = (fmz-self.bin_width/2,fmz+self.bin_width/2)
                    new_ones += 1
                    continue
                fpos = bisect.bisect_right(other_min_mz,fmz)
                fpos -= 1
                if fmz <= other_max_mz[fpos]:
                    self.fmap[f] = other_names[fpos]
                    overlap_match += 1
                else:
                    self.fmap[f] = f
                    self.augmented_features[f] = (fmz-self.bin_width/2,fmz+self.bin_width/2)
                    new_ones += 1
        print "Finished matching ({}). {} exact matches, {} overlap matches, {} new features".format(ftype,exact_match,overlap_match,new_ones)

    def convert(self,dbspectra):
        for doc,spec in dbspectra.items():
            newspec = {}
            for f,i in spec.items():
                newspec[self.fmap[f]] = i
            dbspectra[doc] = newspec
        return dbspectra




def load_mzml_and_make_documents(experiment):
    assert experiment.ms2_file
    peaklist = None
    if experiment.csv_file:
        peaklist = experiment.csv_file.path

    if experiment.experiment_ms2_format == '0':
        loader = LoadMZML(isolation_window=experiment.isolation_window, mz_tol=experiment.mz_tol,
                          rt_tol=experiment.rt_tol, peaklist=peaklist,
                          min_ms1_intensity = experiment.min_ms1_intensity,
                          duplicate_filter = experiment.filter_duplicates,
                          duplicate_filter_mz_tol = experiment.duplicate_filter_mz_tol,
                          duplicate_filter_rt_tol = experiment.duplicate_filter_rt_tol,
                          min_ms1_rt = experiment.min_ms1_rt,
                          max_ms1_rt = experiment.max_ms1_rt,
                          min_ms2_intensity = experiment.min_ms2_intensity,
                          rt_units = experiment.csv_rt_units,
                          mz_col_name = experiment.csv_mz_column,
                          csv_id_col = experiment.csv_id_column,
                          id_field = experiment.ms2_id_field)
    elif experiment.experiment_ms2_format == '1':
        loader = LoadMSP(min_ms1_intensity = experiment.min_ms1_intensity,
                        min_ms2_intensity = experiment.min_ms2_intensity,
                        mz_tol=experiment.mz_tol,
                        rt_tol=experiment.rt_tol,
                        peaklist=peaklist,
                        rt_units = experiment.csv_rt_units,
                        mz_col_name = experiment.csv_mz_column,
                        csv_id_col = experiment.csv_id_column,
                        id_field = experiment.ms2_id_field,
                        name_field = experiment.ms2_name_field)
    elif experiment.experiment_ms2_format == '2':
        loader = LoadMGF(min_ms1_intensity = experiment.min_ms1_intensity,
                        min_ms2_intensity = experiment.min_ms2_intensity,
                        mz_tol=experiment.mz_tol,
                        rt_tol=experiment.rt_tol,
                        peaklist=peaklist,
                        rt_units = experiment.csv_rt_units,
                        mz_col_name = experiment.csv_mz_column,
                        csv_id_col = experiment.csv_id_column,
                        id_field = experiment.ms2_id_field,
                        name_field = experiment.ms2_name_field)

    print "Loading peaks from {} using peaklist {}".format(experiment.ms2_file.path, peaklist)
    ms1, ms2, metadata = loader.load_spectra([experiment.ms2_file.path])
    print "Loaded {} MS1 peaks and {} MS2 peaks".format(len(ms1), len(ms2))


    
    # need to add bin width here..
    bin_widths = {'binned_005':0.005,
                  'binned_01': 0.01,
                  'binned_05': 0.05,
                  'binned_1': 0.1,
                  'binned_5': 0.5}

    bin_width = bin_widths[experiment.featureset.name]
    fm = MakeBinnedFeatures(bin_width = bin_width)
    corpus, word_mz_range = fm.make_features(ms2)
    corpus = corpus[corpus.keys()[0]]

    return corpus, metadata, word_mz_range


def run_lda(corpus, metadata, word_mz_range, K, experiment_id, bin_width = 0.005,n_its=1000,include_motifset = None):
    experiment = Experiment.objects.get(id = experiment_id)
    from motifdb.models import MDBMotifSet,MDBMotif
    if include_motifset:
        import ast
        # convert the string rep of the list into an actual list
        ic = ast.literal_eval(include_motifset)
        motifdb_spectra = {}
        motifdb_metadata = {}
        for mf in ic:
            mset = MDBMotifSet.objects.get(id = mf)
            temp_motifs = MDBMotif.objects.filter(motif_set = mset)
            for m in temp_motifs:
                new_motif_name = "{}(Exp:{})".format(m.name,experiment.id)
                fi = Mass2MotifInstance.objects.filter(mass2motif = m)
                motifdb_spectra[new_motif_name] = {}
                for f in fi:
                    motifdb_spectra[new_motif_name][f.feature.name] = f.probability
                md = jsonpickle.decode(m.metadata)
                motifdb_metadata[new_motif_name] = {}
                for key,value in md.items():
                    motifdb_metadata[new_motif_name][key] = value

        # filter to remove duplicates
        print "Filtering motifs to remove duplicates"
        from motifdb.views import MotifFilter
        mf = MotifFilter(motifdb_spectra,motifdb_metadata)
        motifdb_spectra,motifdb_metadata = mf.filter()

        motifdb_features = set()
        for m,spec in motifdb_spectra.items():
            for f in spec:
                motifdb_features.add(f)

        fm = FeatureMatcher(motifdb_features, word_mz_range, bin_width = bin_width)
        motifdb_spectra = fm.convert(motifdb_spectra)

        # Add the motifdb features to avoid problems when loading the dict into vlda later
        
        added = 0
        for f in motifdb_features:
            if not f in word_mz_range:
                word_mz = float(f.split('_')[1])
                word_mz_min = word_mz - bin_width / 2
                word_mz_max = word_mz + bin_width / 2
                word_mz_range[f] = (word_mz_min, word_mz_max)
                added += 1

        print "Added {} features".format(added)

        vlda = VariationalLDA(corpus, K=K, normalise=1000.0,
                      fixed_topics=motifdb_spectra,
                      fixed_topics_metadata=motifdb_metadata)
        
        vlda.run_vb(n_its=n_its, initialise=True)

        lda_dict = vlda.make_dictionary(metadata=metadata,features=word_mz_range)
        
    else:
        print "Standard with no added motifs"
        vlda = VariationalLDA(corpus=corpus, K=K, normalise=1000.0)
    
        vlda.run_vb(n_its=n_its, initialise=True)

        lda_dict = vlda.make_dictionary(metadata=metadata, features=word_mz_range)
    return lda_dict

