from __future__ import absolute_import, unicode_literals

import logging
import os
import shutil
import sys
import traceback
import pickle
import json

from celery.utils.log import get_task_logger

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_DECOMPOSITION_SOURCE, EXPERIMENT_TYPE
from basicviz.models import Experiment
from decomposition.decomposition_functions import decompose, load_mzml_and_make_documents
from decomposition.models import MotifSet, Decomposition
from load_dict_functions import load_dict
from ms2ldaviz.celery_tasks import app
from .lda_functions import load_mzml_and_make_documents as lda_load_mzml_and_make_documents
from .lda_functions import run_lda


def delete_analysis_dir(exp):

    if exp.ms2_file:
        upload_folder = os.path.dirname(exp.ms2_file.path)
        print 'Deleting %s' % upload_folder
        shutil.rmtree(upload_folder)


# see http://stackoverflow.com/questions/29712938/python-celery-how-to-separate-log-files
# see http://oriolrius.cat/blog/2013/09/06/celery-logs-through-syslog/
def custom_logger(exp):

    experiment_id = 'experiment_%d' % exp.id
    logger = get_task_logger(__name__)
    logger.setLevel(logging.DEBUG)

    upload_folder = os.path.dirname(exp.ms2_file.path)
    media_folder = os.path.dirname(upload_folder)

    # The log files will go to e.g. /Users/joewandy/git/ms2ldaviz/ms2ldaviz/media/logs
    # Create the directory if it doesn't exist
    log_folder = os.path.join(media_folder, 'logs')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    handler = logging.FileHandler(os.path.join(log_folder, experiment_id + '.log'), 'w')
    logger.addHandler(handler)
    return logger


@app.task
def lda_task(exp_id, params):

    exp = Experiment.objects.get(pk=exp_id)
    K = int(params['K'])
    n_its = int(params['n_its'])

    logger = custom_logger(exp)
    logger.info('Running lda on experiment_%d (%s), K=%d' % (exp_id, exp.description, K))
    logger.info('CSV file = %s' % exp.csv_file)
    logger.info('mzML file = %s' % exp.ms2_file)

    # here we redirect all stdout, stderr of this task to our custom logger
    # see http://docs.celeryproject.org/en/latest/userguide/tasks.html#logging
    old_outs = sys.stdout, sys.stderr
    rlevel = app.conf.worker_redirect_stdouts_level

    try:
        app.log.redirect_stdouts_to_logger(logger, rlevel)
        corpus, metadata, word_mz_range = lda_load_mzml_and_make_documents(exp)
        lda_dict = run_lda(corpus, metadata, word_mz_range, K, n_its=n_its)
        feature_set_name = exp.featureset.name
        load_dict(lda_dict, exp, feature_set_name = feature_set_name)

        yes, _ = EXPERIMENT_DECOMPOSITION_SOURCE[1]
        if exp.decomposition_source == yes:
            # TODO: create a MotifSet here from the topics in lda_dict?
            print 'Creating MotifSet'

        ready, _ = EXPERIMENT_STATUS_CODE[1]
        exp.status = ready
        exp.save()

        delete_analysis_dir(exp)

    except:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_outs

@app.task
def decomposition_task(exp_id, params):
    experiment = Experiment.objects.get(pk=exp_id)
    decompose_from = params['decompose_from']

    logger = custom_logger(experiment)
    logger.info('Running decomposition on experiment_%d (%s), decompose_from %s' % (exp_id, experiment.description, decompose_from))
    logger.info('CSV file = %s' % experiment.csv_file)
    logger.info('mzML file = %s' % experiment.ms2_file)

    old_outs = sys.stdout, sys.stderr
    rlevel = app.conf.worker_redirect_stdouts_level
    try:
        app.log.redirect_stdouts_to_logger(logger, rlevel)

        motifset = MotifSet.objects.get(name = decompose_from)
        name = experiment.name + ' decomposition'

        load_mzml_and_make_documents(experiment,motifset)
        decomposition = Decomposition.objects.create(name = name,experiment = experiment,motifset = motifset)
        pending, _ = EXPERIMENT_STATUS_CODE[0]
        decomposition.status = pending
        decomposition.save()

        decompose(decomposition)

        ready, _ = EXPERIMENT_STATUS_CODE[1]
        experiment.status = ready
        experiment.save()
        decomposition.status = ready
        decomposition.save()

        delete_analysis_dir(experiment)

    except:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_outs


@app.task
def just_decompose_task(decomposition_id):

    decomposition = Decomposition.objects.get(id = decomposition_id)

    pending, _ = EXPERIMENT_STATUS_CODE[0]
    decomposition.status = pending
    decomposition.save()

    print "Decomposing {} with {}".format(decomposition.experiment,decomposition.motifset)
    decompose(decomposition)

    ready, _ = EXPERIMENT_STATUS_CODE[1]
    decomposition.status = ready
    decomposition.save()


@app.task
def upload_task(exp_id, params):
    experiment = Experiment.objects.get(pk=exp_id)

    logger = custom_logger(experiment)
    logger.info('Running upload on experiment_%d (%s)' % (exp_id, experiment.description))
    logger.info('dict/json file = %s' % experiment.ms2_file)

    old_outs = sys.stdout, sys.stderr
    rlevel = app.conf.worker_redirect_stdouts_level
    try:
        app.log.redirect_stdouts_to_logger(logger, rlevel)

        filename = params['filename']
        with open(filename, 'r') as f:
            if filename.lower().endswith('.dict'):
                lda_dict = pickle.load(f)
            elif filename.lower().endswith('.json'):
                lda_dict = json.load(f)
            logger.info('Loaded %s' % filename)

            verbose = False
            featureset = params['featureset']
            load_dict(lda_dict, experiment, verbose, featureset)

            ready, _ = EXPERIMENT_STATUS_CODE[1]
            experiment.status = ready
            experiment.save()

            delete_analysis_dir(experiment)

    except:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_outs