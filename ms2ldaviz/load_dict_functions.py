import pickle
import sys

import jsonpickle

from basicviz.models import Experiment,Document,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Alpha
from basicviz.models import BVFeatureSet
from basicviz.views import compute_overlap_score
from ms1analysis.models import Sample, DocSampleIntensity

from django.db import transaction


def add_all_features_set(experiment,features,featureset):
    # Used when we have a dictionary of features with their min and max mz values
    nfeatures = len(features)

    current_features = Feature.objects.filter(featureset = featureset)
    current_names = [f.name for f in current_features]
    ndone = 0
    with transaction.atomic():
        for feature in features:
            # round the name
            # mz = float(feature.split('_')[1])

            if not feature in current_names:
                mz_vals = features[feature]
                f = add_feature_set(feature,featureset)
                f.min_mz = mz_vals[0]
                f.max_mz = mz_vals[1]
                f.save()
            else:
                pass
            ndone+=1
            if ndone % 100 == 0:
                print "Done {} of {}".format(ndone,nfeatures)



def add_all_features(experiment,features):
    # Used when we have a dictionary of features with their min and max mz values
    nfeatures = len(features)
    ndone = 0
    for feature in features:
        mz_vals = features[feature]
        f = add_feature(feature,experiment)
        f.min_mz = mz_vals[0]
        f.max_mz = mz_vals[1]
        f.save()
        ndone += 1
        if ndone % 100 == 0:
            print "Done {} of {}".format(ndone,nfeatures)


def add_document(name,experiment,metadata):
    d = Document.objects.get_or_create(name = name,experiment = experiment,metadata = metadata)[0]
    return d

def add_feature(name,experiment):
    try:
        f = Feature.objects.get_or_create(name = name,experiment = experiment)[0]
    except:
        print name,experiment
        sys.exit(0)
    return f

def add_feature_set(name,featureset):
    try:
        current = Feature.objects.filter(name = name,featureset = featureset)
        if len(current) == 0:
            f = Feature(name = name,featureset = featureset)
            return f
        elif len(current) == 1:
            return current[0]
        elif len(current) > 1:
            print "MULTIPLE FEATURE",name,featureset
            sys.exit(0)
        # f = Feature.objects.get_or_create(name = name,featureset = featureset)[0]
    except:
        print name,featureset
        sys.exit(0)

def add_feature_instance(document,feature,intensity):
    try:
        fi = FeatureInstance.objects.get_or_create(document=document,feature=feature,intensity=intensity)[0]
    except:
        print document,feature,intensity
        sys.exit(0)


def add_topic(topic,experiment,metadata,lda_dict):
    m2m = Mass2Motif.objects.get_or_create(name = topic,experiment = experiment,metadata = metadata)[0]
    for word in lda_dict['beta'][topic]:
        feature = Feature.objects.get(name = word,experiment=experiment)
        Mass2MotifInstance.objects.get_or_create(feature = feature,mass2motif = m2m,probability = lda_dict['beta'][topic][word])[0]
    topic_pos = lda_dict['topic_index'][topic]
    alp = Alpha.objects.get_or_create(mass2motif = m2m,value = lda_dict['alpha'][topic_pos])


def add_topic_set(topic,experiment,metadata,lda_dict,featureset):
    m2m = Mass2Motif.objects.get_or_create(name = topic,experiment = experiment,metadata = metadata)[0]
    for word,probability in lda_dict['beta'][topic].items():
        feature = Feature.objects.get(name = word,featureset = featureset)
        Mass2MotifInstance.objects.get_or_create(feature = feature,mass2motif = m2m,probability = probability)[0]
    topic_pos = lda_dict['topic_index'][topic]
    alp = Alpha.objects.get_or_create(mass2motif = m2m,value = lda_dict['alpha'][topic_pos])


def add_theta(doc_name,experiment,lda_dict):
    document = Document.objects.get(name = doc_name,experiment=experiment)
    for topic in lda_dict['theta'][doc_name]:
        mass2motif = Mass2Motif.objects.get(name = topic,experiment = experiment)
        os = None
        if 'overlap_scores' in lda_dict:
            os = lda_dict['overlap_scores'][doc_name].get(topic,None)
        if os:
            DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc_name][topic],overlap_score = os)[0]
        else:
            DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc_name][topic])[0]

def load_phi(doc_name,experiment,lda_dict):
    document = Document.objects.get(name = doc_name,experiment=experiment)
    for word in lda_dict['phi'][doc_name]:
        feature = Feature.objects.get(name=word,experiment=experiment)
        feature_instance = FeatureInstance.objects.get(document=document,feature=feature)
        for topic in lda_dict['phi'][doc_name][word]:
            mass2motif = Mass2Motif.objects.get(name=topic,experiment=experiment)
            probability = lda_dict['phi'][doc_name][word][topic]
            FeatureMass2MotifInstance.objects.get_or_create(featureinstance = feature_instance,mass2motif = mass2motif,probability = probability)[0]
    

def load_phi_set(doc_name,experiment,lda_dict,featureset):
    document = Document.objects.get(name = doc_name,experiment=experiment)
    for word in lda_dict['phi'][doc_name]:
        feature = Feature.objects.get(name=word,featureset = featureset)
        feature_instance = FeatureInstance.objects.get(document=document,feature=feature)
        for topic in lda_dict['phi'][doc_name][word]:
            mass2motif = Mass2Motif.objects.get(name=topic,experiment=experiment)
            probability = lda_dict['phi'][doc_name][word][topic]
            FeatureMass2MotifInstance.objects.get_or_create(featureinstance = feature_instance,mass2motif = mass2motif,probability = probability)[0]
    

def add_document_words(document,doc_name,experiment,lda_dict):
    for word in lda_dict['corpus'][doc_name]:
        feature = add_feature(word,experiment)
        # feature = Feature.objects.get_or_create(name=word,experiment=experiment)[0]
        # fi = FeatureInstance.objects.get_or_create(document = d,feature = feature, intensity = lda_dict['corpus'][doc][word])
        add_feature_instance(document,feature,lda_dict['corpus'][doc_name][word])

def add_document_words_set(document,doc_name,experiment,lda_dict,featureset):
    for word,intensity in lda_dict['corpus'][doc_name].items():
        feature = Feature.objects.get(name = word,featureset = featureset)
        try:
            fi = FeatureInstance.objects.get_or_create(document = document,feature=feature,intensity = intensity)
        except:
            print document,word,experiment
            sys.exit(0)

def add_sample(sample_name, experiment):
    sample = Sample.objects.get_or_create(name = sample_name, experiment = experiment)[0]
    return sample

def add_doc_sample_intensity(sample, document, intensity):
    i = DocSampleIntensity.objects.get_or_create(sample = sample, document = document, intensity = intensity)[0]

def load_sample_intensity(document, experiment, metadata):
    # metadata = lda_dict['doc_metadata'][doc_name]
    if 'intensities' in metadata:
        for sample_name, intensity in metadata['intensities'].items():
            ## process missing data
            ## if intensity not exist, does not save in database
            if intensity:
                sample = add_sample(sample_name, experiment)
                add_doc_sample_intensity(sample, document, intensity)


def load_dict(lda_dict,experiment,verbose = True,feature_set_name = 'binned_005'):


    # Hard-coded to use the binned 005 featureset
    featureset = BVFeatureSet.objects.get(name = feature_set_name)
    experiment.featureset = featureset
    experiment.save()
    if 'features' in lda_dict:
        print "Explicit feature object: loading them all at once"
        add_all_features_set(experiment,lda_dict['features'],featureset = featureset)


    print "Loading corpus, samples and intensities"
    n_done = 0
    to_do = len(lda_dict['corpus'])
    with transaction.atomic():
        for doc in lda_dict['corpus']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done,to_do)
                experiment.status = "Done {}/{} docs".format(n_done,to_do)
                experiment.save()
            ## remove 'intensities' from metdat before store it into database
            metdat = lda_dict['doc_metadata'][doc].copy()
            metdat.pop('intensities', None)
            metdat = jsonpickle.encode(metdat)
            if verbose:
                print doc,experiment,metdat
            d = add_document(doc,experiment,metdat)
            # d = Document.objects.get_or_create(name=doc,experiment=experiment,metadata=metdat)[0]

            add_document_words_set(d,doc,experiment,lda_dict,featureset)


            load_sample_intensity(d, experiment, lda_dict['doc_metadata'][doc])


            # for word in lda_dict['corpus'][doc]:
            #   feature = add_feature(word,experiment)
            #   # feature = Feature.objects.get_or_create(name=word,experiment=experiment)[0]
            #   # fi = FeatureInstance.objects.get_or_create(document = d,feature = feature, intensity = lda_dict['corpus'][doc][word])
            #   add_feature_instance(d,feature,lda_dict['corpus'][doc][word])
    print "Loading topics"
    n_done = 0
    to_do = len(lda_dict['beta'])
    with transaction.atomic():
        # topic
        m2ms = {}
        for topic in lda_dict['beta']:
            metadata = jsonpickle.encode(lda_dict['topic_metadata'].get(topic, {}))
            m2ms[topic]= Mass2Motif(name=topic, experiment=experiment, metadata=metadata)
        Mass2Motif.objects.bulk_create(m2ms.values())

        # words of topic
        m2mis = []

        features_q = Feature.objects.filter(featureset=featureset)
        features = {r.name: r for r in features_q}
        for topic in lda_dict['beta']:
            m2m = m2ms[topic]
            for word, probability in lda_dict['beta'][topic].items():
                feature = features[word]
                m2mis.append(Mass2MotifInstance(feature=feature, mass2motif=m2m, probability=probability))
        Mass2MotifInstance.objects.bulk_create(m2mis)

        # alphas
        alphas = []
        for topic in lda_dict['beta']:
            m2m = m2ms[topic]
            topic_pos = lda_dict['topic_index'][topic]
            alpha = lda_dict['alpha'][topic_pos]
            alphas.append(Alpha(mass2motif=m2m, value=alpha))
        Alpha.objects.bulk_create(alphas)

    print "Loading theta"
    n_done = 0
    to_do = len(lda_dict['theta'])
    with transaction.atomic():
        for doc in lda_dict['theta']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done,to_do) 
                experiment.status = "Done {}/{} theta".format(n_done,to_do)     
                experiment.save()
            add_theta(doc,experiment,lda_dict)


            # document = Document.objects.get(name = doc,experiment=experiment)
            # for topic in lda_dict['theta'][doc]:
            #   mass2motif = Mass2Motif.objects.get(name = topic,experiment = experiment)
            #   DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc][topic])

    print "Loading phi"
    n_done = 0
    to_do = len(lda_dict['phi'])
    with transaction.atomic():
        for doc in lda_dict['phi']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done,to_do)
                experiment.status = "Done {}/{} phi".format(n_done,to_do)
                experiment.save()
            load_phi_set(doc,experiment,lda_dict,featureset)
            # document = Document.objects.get(name = doc,experiment=experiment)
            # for word in lda_dict['phi'][doc]:
            #   feature = Feature.objects.get(name=word,experiment=experiment)
            #   feature_instance = FeatureInstance.objects.get(document=document,feature=feature)
            #   for topic in lda_dict['phi'][doc][word]:
            #       mass2motif = Mass2Motif.objects.get(name=topic,experiment=experiment)
            #       probability = lda_dict['phi'][doc][word][topic]
            #       FeatureMass2MotifInstance.objects.get_or_create(featureinstance = feature_instance,mass2motif = mass2motif,probability = probability)
    if not 'overlap_scores' in lda_dict:
        print "Computing overlap scores"
        n_done = 0
        dm2ms = DocumentMass2Motif.objects.filter(document__experiment = experiment)
        to_do = len(dm2ms)
        with transaction.atomic():
            for dm2m in dm2ms:
                n_done += 1
                if n_done % 100 == 0:
                    print "Done {}/{}".format(n_done,to_do)
                dm2m.overlap_score = compute_overlap_score(dm2m.mass2motif,dm2m.document)
                dm2m.save()
