from __future__ import absolute_import, unicode_literals

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_DECOMPOSITION_SOURCE
from basicviz.models import Document, Experiment
from decomposition.decomposition_functions import decompose,load_mzml_and_make_documents
from .lda_functions import run_lda
from .lda_functions import load_mzml_and_make_documents as lda_load_mzml_and_make_documents
from decomposition.models import Beta,MotifSet,Decomposition
from ms2ldaviz.celery_tasks import app

# Import the load dict method
from load_dict_functions import load_dict

@app.task
def lda_task(exp_id, params):
    exp = Experiment.objects.get(pk=exp_id)
    K = int(params['K'])
    n_its = int(params['n_its'])
    print 'Running lda on experiment_%d (%s), K=%d' % (exp_id, exp.description, K)
    print 'CSV file = %s' % exp.csv_file
    print 'mzML file = %s' % exp.mzml_file

    corpus, metadata, word_mz_range = lda_load_mzml_and_make_documents(exp)
    lda_dict = run_lda(corpus, metadata, word_mz_range, K, n_its=n_its)
    load_dict(lda_dict, exp)

    yes, _ = EXPERIMENT_DECOMPOSITION_SOURCE[1]
    if exp.decomposition_source == yes:
        # TODO: create a MotifSet here from the topics in lda_dict?
        print 'Creating MotifSet'

    ready, _ = EXPERIMENT_STATUS_CODE[1]
    exp.status = ready
    exp.save()


@app.task
def decomposition_task(exp_id, params):
    experiment = Experiment.objects.get(pk=exp_id)
    decompose_from = params['decompose_from']
    print 'Running decomposition on experiment_%d (%s), decompose_from %s' % (exp_id, experiment.description,
                                                                              decompose_from)
    print 'CSV file = %s' % experiment.csv_file
    print 'mzML file = %s' % experiment.mzml_file

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
