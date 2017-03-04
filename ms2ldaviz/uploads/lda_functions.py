import sys

sys.path.append('../lda/code')
from ms2lda_feature_extraction import LoadMZML, MakeBinnedFeatures
from lda import VariationalLDA

from basicviz.models import Experiment, Document, Feature, FeatureInstance, Mass2Motif, Mass2MotifInstance, \
    DocumentMass2Motif, FeatureMass2MotifInstance, Alpha
import jsonpickle


def load_mzml_and_make_documents(experiment):
    assert experiment.mzml_file
    peaklist = None
    if experiment.csv_file:
        peaklist = experiment.csv_file.path

    loader = LoadMZML(isolation_window=experiment.isolation_window, mz_tol=experiment.mz_tol,
                      rt_tol=experiment.rt_tol, peaklist=peaklist,
                      min_ms1_intensity = experiment.min_ms1_intensity,
                      duplicate_filter = experiment.filter_duplicates,
                      duplicate_filter_mz_tol = experiment.duplicate_filter_mz_tol,
                      duplicate_filter_rt_tol = experiment.duplicate_filter_rt_tol)
    print "Loading peaks from {} using peaklist {}".format(experiment.mzml_file.path, peaklist)
    ms1, ms2, metadata = loader.load_spectra([experiment.mzml_file.path])
    print "Loaded {} MS1 peaks and {} MS2 peaks".format(len(ms1), len(ms2))

    min_ms1_rt = experiment.min_ms1_rt * 60  # seconds
    max_ms1_rt = experiment.max_ms1_rt * 60  # seconds
    min_ms2_intensity = experiment.min_ms2_intensity
    ms1 = filter(lambda x: x.rt > min_ms1_rt and x.rt < max_ms1_rt, ms1)
    ms2 = filter(lambda x: x[3].rt > min_ms1_rt and x[3].rt < max_ms1_rt, ms2)
    ms2 = filter(lambda x: x[2] > min_ms2_intensity, ms2)

    fm = MakeBinnedFeatures()
    corpus, word_mz_range = fm.make_features(ms2)
    corpus = corpus[corpus.keys()[0]]

    return corpus, metadata, word_mz_range


def run_lda(corpus, metadata, word_mz_range, K, n_its=1000):
    vlda = VariationalLDA(corpus=corpus, K=K, normalise=1000.0)
    vlda.run_vb(n_its=n_its, initialise=True)

    lda_dict = vlda.make_dictionary(metadata=metadata, features=word_mz_range)
    return lda_dict

