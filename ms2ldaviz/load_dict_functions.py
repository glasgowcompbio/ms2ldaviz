import pickle

import numpy as np
import sys
import tarfile
import os
from glob import glob

import jsonpickle
# from gensim.models import LdaModel
from tqdm import tqdm

from basicviz.models import Document,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Alpha
from basicviz.models import BVFeatureSet
from basicviz.views import compute_overlap_score
from ms1analysis.models import Sample, DocSampleIntensity

from django.db import transaction, connection


def add_all_features_set(experiment,features,featureset):
    # Used when we have a dictionary of features with their min and max mz values
    nfeatures = len(features)

    current_features = Feature.objects.filter(featureset = featureset).values_list('name', flat=True)
    current_names = {f for f in current_features}
    ndone = 0
    features2add = []
    with transaction.atomic():
        for feature in features:
            # round the name
            # mz = float(feature.split('_')[1])

            if not feature in current_names:
                mz_vals = features[feature]
                f = Feature(name=feature,
                            featureset=featureset,
                            min_mz=mz_vals[0],
                            max_mz=mz_vals[1])
                features2add.append(f)
            else:
                pass
            ndone+=1
            if ndone % 100 == 0:
                print "Done {} of {}".format(ndone,nfeatures)
        Feature.objects.bulk_create(features2add)



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
            assert document.experiment == mass2motif.experiment
            DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc_name][topic],overlap_score = os)[0]            
        else:
            assert document.experiment == mass2motif.experiment
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


    featureset = BVFeatureSet.objects.get(name = feature_set_name)
    experiment.featureset = featureset
    experiment.save()
    if 'features' in lda_dict:
        print "Explicit feature object: loading them all at once"
        add_all_features_set(experiment,lda_dict['features'],featureset = featureset)

    print "Loading corpus, samples and intensities"
    features_q = Feature.objects.filter(featureset=featureset)
    features = {r.name: r for r in features_q}
    n_done = 0
    to_do = len(lda_dict['corpus'])
    doc_dict = {}
    with transaction.atomic():
        for doc in lda_dict['corpus']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done, to_do)
                experiment.status = "Done {}/{} docs".format(n_done, to_do)
                experiment.save()
            ## remove 'intensities' from metdat before store it into database
            metdat = lda_dict['doc_metadata'][doc].copy()
            metdat.pop('intensities', None)
            metdat = jsonpickle.encode(metdat)
            if verbose:
                print doc, experiment, metdat
            doc_dict[doc] = Document(name=doc, experiment=experiment, metadata=metdat)
        Document.objects.bulk_create(doc_dict.values())

        # Now that each document has an row in the database thus an id, add the document children
        feature_instances = {}
        feature_instances_flat = []
        intensities = []
        for doc_id, d in doc_dict.items():
            feature_instances[doc_id] = {}
            # add_document_words_set(d,doc,experiment,lda_dict,featureset)
            for word, intensity in lda_dict['corpus'][doc_id].items():
                fi = FeatureInstance(document=d, feature=features[word], intensity=intensity)
                feature_instances[doc_id][word] = fi
                feature_instances_flat.append(fi)

            # load_sample_intensity(d, experiment, lda_dict['doc_metadata'][doc])
            metdat = lda_dict['doc_metadata'][doc_id]
            if 'intensities' in metdat:
                for sample_name, intensity in metdat['intensities'].items():
                    ## process missing data
                    ## if intensity not exist, does not save in database
                    if intensity:
                        sample = add_sample(sample_name, experiment)
                        # add_doc_sample_intensity(sample, d, intensity)
                        intensities.append(DocSampleIntensity(sample=sample, document=d, intensity=intensity))
        FeatureInstance.objects.bulk_create(feature_instances_flat)
        DocSampleIntensity.objects.bulk_create(intensities)

    print "Loading topics"
    n_done = 0
    to_do = len(lda_dict['beta'])
    with transaction.atomic():
        # topic
        m2ms = {}
        for topic in lda_dict['beta']:
            metadata = jsonpickle.encode(lda_dict['topic_metadata'].get(topic, {}))
            m2ms[topic]= Mass2Motif(name=topic, experiment=experiment, metadata=metadata)
            # m2ms[topic].save()
        Mass2Motif.objects.bulk_create(m2ms.values())
        
        #Do we need this? Seemed to be needed on Simon's local install
        # m2ms = {}
        # for topic in lda_dict['beta']:
        #     m2ms[topic] = Mass2Motif.objects.get(name = topic,experiment = experiment)

        # words of topic
        m2mis = []
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
        docm2ms = []
        for doc in lda_dict['theta']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done,to_do) 
                experiment.status = "Done {}/{} theta".format(n_done,to_do)     
                experiment.save()
            # add_theta(doc,experiment,lda_dict)
            document = doc_dict[doc]
            for topic, prob in lda_dict['theta'][doc].items():
                mass2motif = m2ms[topic]
                os = None
                if 'overlap_scores' in lda_dict:
                    os = lda_dict['overlap_scores'][doc].get(topic, None)
                if os:
                    docm2ms.append(DocumentMass2Motif(document=document, mass2motif=mass2motif,
                                                      probability=prob,
                                                      overlap_score=os))
                else:
                    docm2ms.append(DocumentMass2Motif(document=document, mass2motif=mass2motif,
                                                      probability=prob))
        DocumentMass2Motif.objects.bulk_create(docm2ms)
            # document = Document.objects.get(name = doc,experiment=experiment)
            # for topic in lda_dict['theta'][doc]:
            #   mass2motif = Mass2Motif.objects.get(name = topic,experiment = experiment)
            #   DocumentMass2Motif.objects.get_or_create(document = document,mass2motif = mass2motif,probability = lda_dict['theta'][doc][topic])

    print "Loading phi"
    n_done = 0
    to_do = len(lda_dict['phi'])
    with transaction.atomic():
        phis = []
        for doc_id in lda_dict['phi']:
            n_done += 1
            if n_done % 100 == 0:
                print "Done {}/{}".format(n_done,to_do)
                experiment.status = "Done {}/{} phi".format(n_done,to_do)
                experiment.save()
            #load_phi_set(doc,experiment,lda_dict,featureset)

            for word in lda_dict['phi'][doc_id]:
                feature_instance = feature_instances[doc_id][word]
                for topic, probability in lda_dict['phi'][doc_id][word].items():
                    mass2motif = m2ms[topic]
                    phis.append(FeatureMass2MotifInstance(featureinstance=feature_instance,
                                                          mass2motif=mass2motif, probability=probability))
        FeatureMass2MotifInstance.objects.bulk_create(phis)
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
        dm2ms = DocumentMass2Motif.objects.filter(document__experiment = experiment).select_related('mass2motif', 'document')
        to_do = len(dm2ms)
        with transaction.atomic():
            for dm2m in dm2ms:
                n_done += 1
                if n_done % 100 == 0:
                    print "Done {}/{}".format(n_done,to_do)
                dm2m.overlap_score = compute_overlap_score(dm2m.mass2motif,dm2m.document)
                dm2m.save()


def load_corpus_gensim(experiment, corpus_dict, feature_set_name, ldafile, min_prob_to_keep_beta, min_prob_to_keep_phi,
                       min_prob_to_keep_theta, normalize, verbose):
    """Loads a json and gensim model generated with `run_gensim.py gensim --ldaformat gensim`

    :param experiment: The experiment to add spectra and motifs toa
    :param corpus_dict: JSON dictionary with spectra
    :param feature_set_name: Width of ms2 bins
    :param ldafile: Filename of save gensim model
    :param min_prob_to_keep_beta: Minimum probability to keep beta
    :param min_prob_to_keep_phi: Minimum probability to keep phi
    :param min_prob_to_keep_theta: Minimum probability to keep theta
    :param normalize: Normalize intensities
    :param verbose: If true will show progress bar
    """
    featureset = BVFeatureSet.objects.get(name=feature_set_name)
    corpus, index2doc = build_gensim_corpus(corpus_dict, normalize)
    if 'features' in corpus_dict:
        print("Explicit feature object: loading them all at once")
        add_all_features_set(experiment, corpus_dict['features'], featureset=featureset)
    features_q = Feature.objects.filter(featureset=featureset)
    features = {r.name: r for r in features_q}
    doc_dict = {}
    with transaction.atomic():
        print('Loading mass spectras')
        for doc in tqdm(corpus_dict['corpus'], disable=not verbose):
            # remove 'intensities' from metdat before store it into database
            metdat = corpus_dict['doc_metadata'][doc].copy()
            metdat.pop('intensities', None)
            metdat = jsonpickle.encode(metdat)
            doc_dict[doc] = Document(name=doc, experiment=experiment, metadata=metdat)
        Document.objects.bulk_create(doc_dict.values())
    with transaction.atomic():
        print('Loading peaks')
        # Now that each document has an row in the database thus an id, add the document children
        fi_chunk_size = 1000000
        feature_instances_flat = []
        intensities = []
        for doc_id, d in tqdm(doc_dict.items(), total=len(doc_dict), disable=not verbose):
            # add_document_words_set(d,doc,experiment,lda_dict,featureset)
            for word, intensity in corpus_dict['corpus'][doc_id].items():
                fi = FeatureInstance(document=d, feature=features[word], intensity=intensity)
                feature_instances_flat.append(fi)
                if len(feature_instances_flat) > fi_chunk_size:
                    FeatureInstance.objects.bulk_create(feature_instances_flat)
                    feature_instances_flat = []

            # load_sample_intensity(d, experiment, lda_dict['doc_metadata'][doc])
            metdat = corpus_dict['doc_metadata'][doc_id]
            if 'intensities' in metdat:
                for sample_name, intensity in metdat['intensities'].items():
                    # process missing data
                    # if intensity not exist, does not save in database
                    if intensity:
                        sample = add_sample(sample_name, experiment)
                        # add_doc_sample_intensity(sample, d, intensity)
                        intensities.append(DocSampleIntensity(sample=sample, document=d, intensity=intensity))
        FeatureInstance.objects.bulk_create(feature_instances_flat)
        del feature_instances_flat
        DocSampleIntensity.objects.bulk_create(intensities)
    print('Reading lda gensim file')
    model = LdaModel.load(ldafile)
    # eq to load_dict, but pull directly from gensim model instead of json data struct
    print("Loading Mass2Motif")
    with transaction.atomic():
        m2ms = {}
        topic_metadatas = corpus_dict['topic_metadata']
        if len(topic_metadatas) != model.num_topics:
            # if gensim was run with different num topics than corpus then recompute metadata
            topic_metadatas, topic_index = gen_topic_metadata(model.num_topics)
        for topic_id, topic_metadata in topic_metadatas.items():
            metadata = jsonpickle.encode(topic_metadata)
            m2ms[topic_id] = Mass2Motif(name=topic_id, experiment=experiment, metadata=metadata)
        Mass2Motif.objects.bulk_create(m2ms.values())
    print("Loading Mass2MotifInstance")
    with transaction.atomic():
        m2mis = []
        index2word = {v: k for k, v in corpus_dict['word_index'].items()}
        for tid, topic in tqdm(enumerate(model.get_topics()), total=model.num_topics, disable=not verbose):
            topic = topic / topic.sum()  # normalize to probability distribution
            topic_id = 'motif_{0}'.format(tid)
            m2m = m2ms[topic_id]
            for idx in np.argsort(-topic):
                if topic[idx] > min_prob_to_keep_beta:
                    feature = features[index2word[idx]]
                    probability = float(topic[idx])
                    m2mis.append(Mass2MotifInstance(feature=feature, mass2motif=m2m, probability=probability))
        Mass2MotifInstance.objects.bulk_create(m2mis)
        del m2mis
    print("Loading Alpha")
    with transaction.atomic():
        alphas = []
        for tid, alpha in tqdm(enumerate(model.alpha), total=model.num_topics, disable=not verbose):
            topic_id = 'motif_{0}'.format(tid)
            m2m = m2ms[topic_id]
            alphas.append(Alpha(mass2motif=m2m, value=alpha))
        Alpha.objects.bulk_create(alphas)
    print("Loading theta")
    with transaction.atomic():
        docm2ms = []
        for doc_id, bow in tqdm(enumerate(corpus), total=len(corpus), disable=not verbose):
            document = doc_dict[index2doc[doc_id]]
            topics = model.get_document_topics(bow, minimum_probability=min_prob_to_keep_theta)
            for tid, prob in topics:
                topic_id = 'motif_{0}'.format(tid)
                mass2motif = m2ms[topic_id]
                docm2ms.append(DocumentMass2Motif(document=document, mass2motif=mass2motif,
                                                  probability=float(prob)))
        DocumentMass2Motif.objects.bulk_create(docm2ms)
        del docm2ms
    print("Loading phi")
    with transaction.atomic():
        phis = []
        phi_chunk_size = 100000
        corpus_topics = model.get_document_topics(corpus, per_word_topics=True,
                                                  minimum_probability=min_prob_to_keep_theta,
                                                  minimum_phi_value=min_prob_to_keep_phi)
        # corpus_topics is array of [topic_theta, topics_per_word,topics_per_word_phi] for each document
        for doc_id, doc_topics in tqdm(enumerate(corpus_topics), total=len(corpus), disable=not verbose):
            topics_per_word_phi = doc_topics[2]

            doc_name = index2doc[doc_id]
            doc = doc_dict[doc_name]
            bow = corpus[doc_id]
            word_intens = {k: v for k, v in bow}
            feature_instances = {r.feature.name: r for r in
                                 FeatureInstance.objects.filter(document=doc).select_related('feature')}
            for word_id, topics in topics_per_word_phi:
                feature_instance = feature_instances[index2word[word_id]]
                for tid, phi in topics:
                    topic_id = 'motif_{0}'.format(tid)
                    mass2motif = m2ms[topic_id]
                    probability = phi / word_intens[word_id]
                    phis.append(FeatureMass2MotifInstance(featureinstance=feature_instance,
                                                          mass2motif=mass2motif, probability=probability))
            if len(phis) > phi_chunk_size:
                # Flush phis to db, to prevent big memory footprint
                FeatureMass2MotifInstance.objects.bulk_create(phis)
                phis = []
        FeatureMass2MotifInstance.objects.bulk_create(phis)
        del phis
    if 'overlap_scores' not in corpus_dict:
        print("Computing overlap scores")
        # Compute overlap score with sql
        with connection.cursor() as cursor:
            """
            SELECT * FROM basicviz_documentmass2motif WHERE id=149332;
               id   |    probability     | document_id | mass2motif_id | validated |   overlap_score
            --------+--------------------+-------------+---------------+-----------+--------------------
             149332 | 0.0169265381991863 |       76297 |          1150 |           | 0.0275374519023851

            SELECT * FROM
            basicviz_featureinstance fi
            WHERE fi.document_id = 76297;

            SELECT * FROM
            basicviz_featuremass2motifinstance
            WHERE featureinstance_id IN (
             SELECT id FROM
                basicviz_featureinstance fi
                WHERE fi.document_id = 76297
            );

            SELECT sum(fmmi.probability * mmi.probability) FROM
            basicviz_featuremass2motifinstance fmmi
            JOIN basicviz_featureinstance fi ON fmmi.featureinstance_id = fi.id
            JOIN basicviz_mass2motifinstance mmi
                ON  mmi.feature_id = fi.feature_id
            WHERE fi.document_id = 76297
            AND mmi.mass2motif_id = 1150
            ;
            // Score matches

            SELECT dmm.* FROM basicviz_documentmass2motif dmm
            JOIN basicviz_document d ON d.id = dmm.document_id
            WHERE experiment_id = 6;
               id   |    probability     | document_id | mass2motif_id | validated |    overlap_score
            --------+--------------------+-------------+---------------+-----------+----------------------
             150337 | 0.0131358103826642 |       76430 |          1152 |           |    0.142929254448449
             150234 | 0.0100447973236442 |       76511 |           962 |           |                    0

            SELECT sum(fmmi.probability * mmi.probability) FROM
            basicviz_featuremass2motifinstance fmmi
            JOIN basicviz_featureinstance fi ON fmmi.featureinstance_id = fi.id
            JOIN basicviz_mass2motifinstance mmi
                ON  mmi.feature_id = fi.feature_id
            WHERE fi.document_id = 76430
            AND mmi.mass2motif_id = 1152
            ;
            // Score matches

            SELECT sum(fmmi.probability * mmi.probability) FROM
            basicviz_featuremass2motifinstance fmmi
            JOIN basicviz_featureinstance fi ON fmmi.featureinstance_id = fi.id
            JOIN basicviz_mass2motifinstance mmi
                ON  mmi.feature_id = fi.feature_id
            WHERE fi.document_id = 76511
            AND mmi.mass2motif_id = 962
            ;
            // No rows, matches score

            SELECT
                dmm.id, sum(mmi.probability * fmmi.probability) AS overlap_score2,
                dmm.overlap_score
            FROM basicviz_document d
            JOIN basicviz_documentmass2motif dmm ON dmm.document_id=d.id
            JOIN basicviz_featureinstance fi ON fi.document_id=dmm.document_id
            JOIN basicviz_featuremass2motifinstance fmmi ON fmmi.featureinstance_id=fi.id
            JOIN basicviz_mass2motifinstance mmi ON mmi.feature_id = fi.feature_id AND mmi.mass2motif_id=dmm.mass2motif_id
            WHERE d.experiment_id = 6
            AND dmm.document_id = 76297
            AND dmm.mass2motif_id = 1150
            GROUP BY dmm.id
            ;
            // Score matches

            SELECT * FROM (
                SELECT
                    dmm.id, sum(mmi.probability * fmmi.probability) AS overlap_score2,
                    dmm.overlap_score
                FROM basicviz_document d
                JOIN basicviz_documentmass2motif dmm ON dmm.document_id=d.id
                JOIN basicviz_featureinstance fi ON fi.document_id=dmm.document_id
                JOIN basicviz_featuremass2motifinstance fmmi ON fmmi.featureinstance_id=fi.id
                JOIN basicviz_mass2motifinstance mmi ON mmi.feature_id = fi.feature_id AND mmi.mass2motif_id=dmm.mass2motif_id
                WHERE d.experiment_id = 6
                GROUP BY dmm.id
            ) a WHERE a.id IN (150337, 150234, 149332);
            // Scores matches
            """

            # when db changes this query can fail
            overlap_score_sql = """UPDATE
                basicviz_documentmass2motif t
            SET
                overlap_score = a.overlap_score
            FROM (
                SELECT
                    dmm.id, sum(mmi.probability * fmmi.probability) AS overlap_score
                FROM basicviz_document d
                JOIN basicviz_documentmass2motif dmm ON dmm.document_id=d.id
                JOIN basicviz_featureinstance fi ON fi.document_id=dmm.document_id
                JOIN basicviz_featuremass2motifinstance fmmi ON fmmi.featureinstance_id=fi.id
                JOIN basicviz_mass2motifinstance mmi ON mmi.feature_id = fi.feature_id AND mmi.mass2motif_id=dmm.mass2motif_id
                WHERE d.experiment_id = %s
                GROUP BY dmm.id
            ) a
            WHERE t.id = a.id
            """
            cursor.execute(overlap_score_sql, [experiment.id])


def prepare_gensim_archive(fn):
    """Unpacks a gensim archive file

    :param fn: Filename of gensim archive file
    :return: Filename of an on disk gensim model which can be loaded into memory with LdaModel.load()
    """
    if fn.endswith('.tar') or fn.endswith('.bz2') or fn.endswith('.gz'):
        upload_folder = os.path.dirname(fn)
        with tarfile.open(fn) as tar:
            tar.extractall(upload_folder)
        gensim_model_extension = '.expElogbeta.npy'
        return glob(os.path.join(upload_folder, '*' + gensim_model_extension))[0].replace(gensim_model_extension, '')
    else:
        raise NotImplementedError('Unknown archive format')


def build_gensim_corpus(lda_dict, normalize):
    corpus = []
    index2doc = []
    for doc in sorted(lda_dict['corpus'].keys()):
        words = lda_dict['corpus'][doc]
        bow = []
        max_score = max(words.values())
        for word in sorted(words.keys()):
            score = words[word]
            normalized_score = score * normalize / max_score
            bow.append((lda_dict['word_index'][word], normalized_score))
        corpus.append(bow)
        index2doc.append(doc)
    return corpus, index2doc


def gen_topic_metadata(k):
    """Topic metadata generation copied from lda/lda.py:VariationalLDA:__init__:229-233

    Sets fixed topics to 0
    """
    topic_index = {}
    topic_metadata = {}
    for topic_pos in range(0, k):
        topic_name = 'motif_{}'.format(topic_pos)
        topic_index[topic_name] = topic_pos
        topic_metadata[topic_name] = {'name': topic_name, 'type': 'learnt'}
    return topic_metadata, topic_index