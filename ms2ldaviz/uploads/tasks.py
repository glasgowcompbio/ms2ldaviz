from __future__ import absolute_import, unicode_literals

from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Document, Experiment
from decomposition.decomposition_functions import decompose,load_mzml_and_make_documents
from .lda_functions import run_lda, load_dict
from .lda_functions import load_mzml_and_make_documents as lda_load_mzml_and_make_documents
from decomposition.models import Beta,MotifSet,Decomposition
from ms2ldaviz.celery_tasks import app


@app.task
def lda_task(exp_id, params):
    exp = Experiment.objects.get(pk=exp_id)
    K = int(params['K'])
    print 'Running lda on experiment_%d (%s), K=%d' % (exp_id, exp.description, K)
    print 'CSV file = %s' % exp.csv_file
    print 'mzML file = %s' % exp.mzml_file

    corpus, metadata, word_mz_range = lda_load_mzml_and_make_documents(exp)
    lda_dict = run_lda(corpus, metadata, word_mz_range, K, n_its=1000)
    load_dict(lda_dict, exp)

    ready, desc = EXPERIMENT_STATUS_CODE[1]
    exp.status = ready
    exp.save()


@app.task
def decomposition_task(exp_id, params):
    experiment = Experiment.objects.get(pk=exp_id)
    decompose_from = params['decompose_from'] # This is not used - user should choose a MOTIFSET
    print 'Running decomposition on experiment_%d (%s), decompose_from %s' % (exp_id, experiment.description,
                                                                              decompose_from)
    print 'CSV file = %s' % experiment.csv_file
    print 'mzML file = %s' % experiment.mzml_file

    # These two should come from the form...
    motifset = MotifSet.objects.get(name = 'massbank_motifset')
    name = experiment.name + ' decomposition'

    load_mzml_and_make_documents(experiment,motifset)
    decomposition = Decomposition.objects.create(name = name,experiment = experiment,motifset = motifset)
    decompose(decomposition)

    ready, desc = EXPERIMENT_STATUS_CODE[1]
    experiment.status = ready
    experiment.save()

