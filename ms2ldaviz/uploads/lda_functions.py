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
                      rt_tol=experiment.rt_tol, peaklist=peaklist)
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
    vlda = VariationalLDA(corpus=corpus, K=300, normalise=1000.0)
    vlda.run_vb(n_its=n_its, initialise=True)

    lda_dict = vlda.make_dictionary(metadata=metadata, features=word_mz_range)
    return lda_dict


def load_dict(lda_dict, experiment):
    print "Loading corpus"
    n_done = 0
    to_do = len(lda_dict['corpus'])
    for doc in lda_dict['corpus']:
        n_done += 1
        if n_done % 100 == 0:
            print "Done {}/{}".format(n_done, to_do)
            experiment.status = "Done {}/{} docs".format(n_done, to_do)
            experiment.save()
        metdat = jsonpickle.encode(lda_dict['doc_metadata'][doc])
        print doc, experiment, metdat
        d = add_document(doc, experiment, metdat)
        add_document_words(d, doc, experiment, lda_dict)

    print "Loading topics"
    n_done = 0
    to_do = len(lda_dict['beta'])
    for topic in lda_dict['beta']:
        n_done += 1
        if n_done % 100 == 0:
            print "Done {}/{}".format(n_done, to_do)
            experiment.status = "Done {}/{} topics".format(n_done, to_do)
            experiment.save()
        metadata = {}
        metadata = lda_dict['topic_metadata'].get(topic, {})
        add_topic(topic, experiment, jsonpickle.encode(metadata), lda_dict)

    print "Loading theta"
    n_done = 0
    to_do = len(lda_dict['theta'])
    for doc in lda_dict['theta']:
        n_done += 1
        if n_done % 100 == 0:
            print "Done {}/{}".format(n_done, to_do)
            experiment.status = "Done {}/{} theta".format(n_done, to_do)
            experiment.save()
        add_theta(doc, experiment, lda_dict)

    print "Loading phi"
    n_done = 0
    to_do = len(lda_dict['phi'])
    for doc in lda_dict['phi']:
        n_done += 1
        if n_done % 100 == 0:
            print "Done {}/{}".format(n_done, to_do)
            experiment.status = "Done {}/{} phi".format(n_done, to_do)
            experiment.save()
        load_phi(doc, experiment, lda_dict)
    experiment.status = 'all loaded'
    experiment.save()


def add_all_features(experiment, features):
    # Used when we have a dictionary of features with their min and max mz values
    nfeatures = len(features)
    ndone = 0
    for feature in features:
        mz_vals = features[feature]
        f = add_feature(feature, experiment)
        f.min_mz = mz_vals[0]
        f.max_mz = mz_vals[1]
        f.save()
        ndone += 1
        if ndone % 100 == 0:
            print "Done {} of {}".format(ndone, nfeatures)


def add_document(name, experiment, metadata):
    d = Document.objects.get_or_create(name=name, experiment=experiment, metadata=metadata)[0]
    return d


def add_feature(name, experiment):
    try:
        f = Feature.objects.get_or_create(name=name, experiment=experiment)[0]
    except:
        print name, experiment
        sys.exit(0)
    return f


def add_feature_instance(document, feature, intensity):
    try:
        fi = FeatureInstance.objects.get_or_create(document=document, feature=feature, intensity=intensity)[0]
    except:
        print document, feature, intensity
        sys.exit(0)


def add_topic(topic, experiment, metadata, lda_dict):
    m2m = Mass2Motif.objects.get_or_create(name=topic, experiment=experiment, metadata=metadata)[0]
    for word in lda_dict['beta'][topic]:
        feature = Feature.objects.get(name=word, experiment=experiment)
        Mass2MotifInstance.objects.get_or_create(feature=feature, mass2motif=m2m,
                                                 probability=lda_dict['beta'][topic][word])[0]
    topic_pos = lda_dict['topic_index'][topic]
    alp = Alpha.objects.get_or_create(mass2motif=m2m, value=lda_dict['alpha'][topic_pos])


def add_theta(doc_name, experiment, lda_dict):
    document = Document.objects.get(name=doc_name, experiment=experiment)
    for topic in lda_dict['theta'][doc_name]:
        mass2motif = Mass2Motif.objects.get(name=topic, experiment=experiment)
        DocumentMass2Motif.objects.get_or_create(document=document, mass2motif=mass2motif,
                                                 probability=lda_dict['theta'][doc_name][topic])[0]


def load_phi(doc_name, experiment, lda_dict):
    document = Document.objects.get(name=doc_name, experiment=experiment)
    for word in lda_dict['phi'][doc_name]:
        feature = Feature.objects.get(name=word, experiment=experiment)
        feature_instance = FeatureInstance.objects.get(document=document, feature=feature)
        for topic in lda_dict['phi'][doc_name][word]:
            mass2motif = Mass2Motif.objects.get(name=topic, experiment=experiment)
            probability = lda_dict['phi'][doc_name][word][topic]
            FeatureMass2MotifInstance.objects.get_or_create(featureinstance=feature_instance, mass2motif=mass2motif,
                                                            probability=probability)[0]


def add_document_words(document, doc_name, experiment, lda_dict):
    for word in lda_dict['corpus'][doc_name]:
        feature = add_feature(word, experiment)
        add_feature_instance(document, feature, lda_dict['corpus'][doc_name][word])
