#!/usr/bin/env python
import textwrap
from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
import json
import os
import sys

from tqdm import tqdm

# Never let numpy use more than one core, otherwise each worker of LdaMulticore will use all cores for numpy
# TODO allow multicore usage when not running LdaMulticore

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append('../lda/code')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from ms2lda_feature_extraction import LoadMZML, MakeBinnedFeatures, LoadMSP, LoadMGF
from basicviz.models import Experiment, User, BVFeatureSet, UserExperiment, JobLog
from load_dict_functions import load_dict, load_corpus_gensim, build_gensim_corpus
from lda import VariationalLDA
# from gensim.models.ldamulticore import LdaMulticore
# from gensim.models.ldamodel import LdaModel
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def build_parser():
    parser = ArgumentParser(description="Run gensim lda on MS2 file and insert into db", epilog=textwrap.dedent("""
    
    run_gensim.py corpus bla.msp corpus.json
    run_gensim.py gensim corpus.json lda.json
    run_gensim.py insert lda.json

    Or piped
    
    run_gensim.py corpus bla.msp - | run_gensim.py gensim - - | run_gensim.py insert -
    
    """), formatter_class=RawDescriptionHelpFormatter)
    sc = parser.add_subparsers(dest='subcommand')

    # corpus
    corpus = sc.add_parser('corpus', help='Generate corpus/features from MS2 file')
    corpus.add_argument('ms2_file', help="MS2 file")
    corpus.add_argument('corpusjson', type=FileType('w'), help="corpus file")
    corpus.add_argument('-f', '--ms2_format', default='msp', help='Format of MS2 file', choices=('msp', 'mgf', 'mzxml'))
    corpus.add_argument('--min_ms1_intensity', type=float, default=0.0, help='Minimum intensity of MS1 peaks to store  (default: %(default)s)')
    corpus.add_argument('--min_ms2_intensity', type=float, default=5000.0, help='Minimum intensity of MS2 peaks to store  (default: %(default)s)')
    corpus.add_argument('--mz_tol', type=float, default=5.0, help='Mass tolerance when linking peaks from the peaklist to those found in MS2 file (ppm) (default: %(default)s)')
    corpus.add_argument('--rt_tol', type=float, default=10.0, help='Retention time tolerance when linking peaks from the peaklist to those found in MS2 file (seconds)  (default: %(default)s)')
    corpus.add_argument('-k', type=int, default=300, help='Number of topics (default: %(default)s)')
    corpus.add_argument('--feature_set_name', default='binned_005',
                        choices=('binned_1', 'binned_01', 'binned_5', 'binned_005', 'binned_05'),
                        help='Choose width of ms2 bins')
    corpus.set_defaults(func=msfile2corpus)

    # lda
    lda = sc.add_parser('gensim', help='Run lda using gensim')
    lda.add_argument('corpusjson', type=FileType('r'), help="corpus file")
    lda.add_argument('ldafile', help="lda file")
    lda.add_argument('-k', type=int, default=300, help='Number of topics (default: %(default)s)')
    lda.add_argument('-n', type=int, default=50, help='Number of iterations (default: %(default)s)')
    lda.add_argument('--gamma_threshold', default=0.001, type=float, help='Minimum change in the value of the gamma parameters to continue iterating (default: %(default)s)')
    lda.add_argument('--chunksize', default=2000, type=int, help='Number of documents to be used in each training chunk, use 0 for same size as corpus (default: %(default)s)')
    lda.add_argument('--batch', action='store_true', help='When set will use batch learning otherwise online learning (default: %(default)s)')
    lda.add_argument('--normalize', type=int, default=1000, help='Normalize intensities (default: %(default)s)')
    lda.add_argument('--passes', type=int, default=1, help='Number of passes through the corpus during training (default: %(default)s)')
    lda.add_argument('--min_prob_to_keep_beta', type=float, default=1e-3, help='Minimum probability to keep beta (default: %(default)s)')
    lda.add_argument('--min_prob_to_keep_phi', type=float, default=1e-2, help='Minimum probability to keep phi (default: %(default)s)')
    lda.add_argument('--min_prob_to_keep_theta', type=float, default=1e-2, help='Minimum probability to keep theta (default: %(default)s)')
    lda.add_argument('--alpha', default='symmetric', choices=('asymmetric', 'symmetric'), help="Prior selecting strategies (default: %(default)s)")
    lda.add_argument('--eta', help='Can be a float or "auto". Default is (default: %(default)s)')
    lda.add_argument('--workers', type=int, default=4, help='Number of workers. 0 will use single core LdaCore otherwise will use LdaMulticore (default: %(default)s)')
    lda.add_argument('--random_seed', type=int, help='Random seed to use, Useful for reproducibility. (default: %(default)s)')
    lda.add_argument('--ldaformat', default='json', choices=('json', 'gensim'), help='Store lda model in json or jensim format')
    lda.set_defaults(func=gensim)

    # insert
    insert = sc.add_parser('insert', help='Insert lda result into db')
    insert.add_argument('ldajson', type=FileType('r'), help="lda file")
    insert.add_argument('owner', help='Experiment owner')
    insert.add_argument('experiment', help='Experiment name')
    insert.add_argument('--description', default='')
    insert.add_argument('--featureset', default='binned_005',
                        choices=('binned_1', 'binned_01', 'binned_5', 'binned_005', 'binned_05'),
                        help='Choose width of ms2 bins')
    insert.set_defaults(func=insert_lda)

    # insert
    insert_gensim = sc.add_parser('insert_gensim', help='Insert gensim lda result into db')
    insert_gensim.add_argument('corpusjson', type=FileType('r'), help="corpus file")
    insert_gensim.add_argument('ldafile', help="lda gensim file")
    insert_gensim.add_argument('owner', help='Experiment owner')
    insert_gensim.add_argument('experiment', help='Experiment name')
    insert_gensim.add_argument('--description', default='')
    insert_gensim.add_argument('--normalize', type=int, default=1000, help='Normalize intensities (default: %(default)s)')
    insert_gensim.add_argument('--min_prob_to_keep_beta', type=float, default=1e-3, help='Minimum probability to keep beta (default: %(default)s)')
    insert_gensim.add_argument('--min_prob_to_keep_phi', type=float, default=1e-2, help='Minimum probability to keep phi (default: %(default)s)')
    insert_gensim.add_argument('--min_prob_to_keep_theta', type=float, default=1e-2, help='Minimum probability to keep theta (default: %(default)s)')
    insert_gensim.add_argument('--feature_set_name', default='binned_005',
                               choices=('binned_1', 'binned_01', 'binned_5', 'binned_005', 'binned_05'),
                               help='Choose width of ms2 bins')
    insert_gensim.set_defaults(func=insert_gensim_lda)

    return parser


def msfile2corpus(ms2_file, ms2_format,
                  min_ms1_intensity, min_ms2_intensity,
                  mz_tol, rt_tol,
                  feature_set_name,
                  k,
                  corpusjson):
    if ms2_format == 'mzxml':
        loader = LoadMZML(mz_tol=mz_tol,
                          rt_tol=rt_tol, peaklist=None,
                          min_ms1_intensity=min_ms1_intensity,
                          min_ms2_intensity=min_ms2_intensity)
    elif ms2_format == 'msp':
        loader = LoadMSP(min_ms1_intensity=min_ms1_intensity,
                         min_ms2_intensity=min_ms2_intensity,
                         mz_tol=mz_tol,
                         rt_tol=rt_tol,
                         peaklist=None,
                         name_field="")
    elif ms2_format == 'mgf':
        loader = LoadMGF(min_ms1_intensity=min_ms1_intensity,
                         min_ms2_intensity=min_ms2_intensity,
                         mz_tol=mz_tol,
                         rt_tol=rt_tol,
                         peaklist=None,
                         name_field="")
    else:
        raise NotImplementedError('Unknown ms2 format')
    ms1, ms2, metadata = loader.load_spectra([ms2_file])

    bin_widths = {'binned_005': 0.005,
                  'binned_01': 0.01,
                  'binned_05': 0.05,
                  'binned_1': 0.1,
                  'binned_5': 0.5}

    bin_width = bin_widths[feature_set_name]

    fm = MakeBinnedFeatures(bin_width=bin_width)
    corpus, features = fm.make_features(ms2)
    corpus = corpus[corpus.keys()[0]]

    # To insert in db some additional data is generated inVariationalLDA
    vlda = VariationalLDA(corpus=corpus, K=k)
    lda_dict = {'corpus': corpus,
                'word_index': vlda.word_index,
                'doc_index': vlda.doc_index,
                'doc_metadata': metadata,
                'topic_index': vlda.topic_index,
                'topic_metadata': vlda.topic_metadata,
                'features': features
                }
    json.dump(lda_dict, corpusjson)


def compute_overlap_scores(lda_dictionary):
    # Compute the overlap scores for the lda model in dictionary format
    overlap_scores = {}
    for doc, phi in lda_dictionary['phi'].items():
        motifs = lda_dictionary['theta'][doc].keys()
        doc_overlaps = {m : 0.0 for m in motifs}
        for word, probs in phi.items():
            for m in motifs:
                if word in lda_dictionary['beta'][m] and m in probs:
                    doc_overlaps[m] += lda_dictionary['beta'][m][word]*probs[m]
        overlap_scores[doc] = {}
        for m in doc_overlaps:
            overlap_scores[doc][m] = doc_overlaps[m]
    return overlap_scores


def gensim(corpusjson, ldafile,
           n, k, gamma_threshold,
           chunksize, batch, normalize, passes,
           min_prob_to_keep_beta, min_prob_to_keep_phi, min_prob_to_keep_theta,
           alpha, eta, workers, random_seed,
           ldaformat
           ):

    logging.warning('Reading corpus json')
    if eta is not None and eta != 'auto':
        eta = float(eta)
    lda_dict = json.load(corpusjson)
    corpus, index2doc = build_gensim_corpus(lda_dict, normalize)
    if chunksize == 0:
        chunksize = len(corpus)

    logging.warning('Start lda')
    if workers > 0:
        lda = LdaMulticore(corpus,
                           num_topics=k, iterations=n,
                           per_word_topics=True, gamma_threshold=gamma_threshold,
                           chunksize=chunksize, batch=batch,
                           passes=passes, alpha=alpha, eta=eta,
                           workers=workers,
                           random_state=random_seed,
                           dtype=np.float64,
                           )
    else:
        lda = LdaModel(corpus,
                       num_topics=k, iterations=n,
                       per_word_topics=True, gamma_threshold=gamma_threshold,
                       chunksize=chunksize, update_every=0 if batch else 1,
                       passes=passes, alpha=alpha, eta=eta,
                       random_state=random_seed,
                       dtype=np.float64,
                       )

    if ldaformat == 'gensim':
        logging.warning('Saving gensim to disk')
        lda.save(ldafile)
        return

    logging.warning('Build beta matrix')
    beta = {}
    index2word = {v: k for k, v in lda_dict['word_index'].items()}
    for tid, topic in tqdm(enumerate(lda.get_topics()), total=k):
        topic = topic / topic.sum()  # normalize to probability distribution
        beta['motif_{0}'.format(tid)] = {index2word[idx]: float(topic[idx]) for idx in np.argsort(-topic) if
                                         topic[idx] > min_prob_to_keep_beta}

    logging.warning('Build theta matrix')
    theta = {}
    for doc_id, bow in tqdm(enumerate(corpus), total=len(corpus)):
        topics = lda.get_document_topics(bow, minimum_probability=min_prob_to_keep_theta)
        theta[index2doc[doc_id]] = {'motif_{0}'.format(topic_id): float(prob) for topic_id, prob in topics}

    logging.warning('Build phi matrix')
    phis = {}
    corpus_topics = lda.get_document_topics(corpus, per_word_topics=True,
                                            minimum_probability=min_prob_to_keep_theta,
                                            minimum_phi_value=min_prob_to_keep_phi)
    # corpus_topics is array of [topic_theta, topics_per_word,topics_per_word_phi] for each document
    for doc_id, doc_topics in tqdm(enumerate(corpus_topics), total=len(corpus)):
        topics_per_word_phi = doc_topics[2]
        doc_name = index2doc[doc_id]
        word_intens = {k: v for k, v in corpus[doc_id]}
        phis[doc_name] = {
            index2word[word_id]: {
                'motif_{0}'.format(topic_id): phi / word_intens[word_id] for topic_id, phi in topics
            } for word_id, topics in topics_per_word_phi}

    logging.warning('Build alpha matrix')
    lda_dict['alpha'] = [float(d) for d in lda.alpha]
    lda_dict['beta'] = beta
    lda_dict['theta'] = theta
    lda_dict['phi'] = phis
    lda_dict['K'] = k
    logging.warning('Build overlap_scores matrix')
    lda_dict['overlap_scores'] = compute_overlap_scores(lda_dict)
    logging.warning('Build json matrix')
    with open(ldafile, 'w') as f:
        json.dump(lda_dict, f)


def insert_gensim_lda(corpusjson, ldafile, experiment, owner, description, normalize, min_prob_to_keep_beta,
                      min_prob_to_keep_theta, min_prob_to_keep_phi, feature_set_name):
    featureset, new_experiment = create_experiment(description, experiment, owner, feature_set_name)
    verbose = True

    print('Reading corpus json file')
    corpus_dict = json.load(corpusjson)
    load_corpus_gensim(new_experiment, corpus_dict, feature_set_name, ldafile, min_prob_to_keep_beta, min_prob_to_keep_phi,
                       min_prob_to_keep_theta, normalize, verbose)

    # Done inserting
    new_experiment.status = '1'
    new_experiment.save()


def insert_lda(ldajson, experiment, owner, description, feature_set_name):
    featureset, new_experiment = create_experiment(description, experiment, owner, feature_set_name)

    ldajson = open(ldajson)
    lda_dict = json.load(ldajson)
    load_dict(lda_dict, new_experiment, feature_set_name=featureset.name)

    new_experiment.status = '1'
    new_experiment.save()


def create_experiment(description, experiment, owner, feature_set_name):
    user = User.objects.get(username=owner)
    featureset = BVFeatureSet.objects.get(name=feature_set_name)
    new_experiment = Experiment(name=experiment,
                                description=description,
                                status='0',
                                experiment_type='0',
                                featureset=featureset)
    new_experiment.save()
    UserExperiment.objects.create(user=user, experiment=new_experiment, permission='edit')
    JobLog.objects.create(user=user, experiment=new_experiment,
                          tasktype='Uploaded and ' + new_experiment.experiment_type)
    return featureset, new_experiment


def main(argv=sys.argv[1:]):
    parser = build_parser()
    args = parser.parse_args(argv)
    fargs = vars(args)
    if 'func' in fargs:
        func = args.func
        del (fargs['subcommand'])
        del (fargs['func'])
        func(**fargs)
    else:
        if 'subcommand' in args:
            parser.parse_args([args.subcommand, '--help'])
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
