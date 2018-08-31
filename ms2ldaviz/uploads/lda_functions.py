import sys

sys.path.append('../lda/code')
from ms2lda_feature_extraction import LoadMZML, MakeBinnedFeatures, LoadMSP, LoadMGF
from lda import VariationalLDA

from basicviz.models import Experiment, Document, Feature, FeatureInstance, Mass2Motif, Mass2MotifInstance, \
    DocumentMass2Motif, FeatureMass2MotifInstance, Alpha
import jsonpickle


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
                        id_field = experiment.ms2_id_field)
    elif experiment.experiment_ms2_format == '2':
        loader = LoadMGF(min_ms1_intensity = experiment.min_ms1_intensity,
                        min_ms2_intensity = experiment.min_ms2_intensity,
                        mz_tol=experiment.mz_tol,
                        rt_tol=experiment.rt_tol,
                        peaklist=peaklist,
                        rt_units = experiment.csv_rt_units,
                        mz_col_name = experiment.csv_mz_column,
                        csv_id_col = experiment.csv_id_column,
                        id_field = experiment.ms2_id_field)

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


def run_lda(corpus, metadata, word_mz_range, K, n_its=1000):
    vlda = VariationalLDA(corpus=corpus, K=K, normalise=1000.0)
    vlda.run_vb(n_its=n_its, initialise=True)

    lda_dict = vlda.make_dictionary(metadata=metadata, features=word_mz_range)
    return lda_dict

