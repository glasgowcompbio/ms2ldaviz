from __future__ import absolute_import, unicode_literals

from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Document, Experiment
from decomposition.decomposition_functions import load_mzml_and_make_documents, decompose
from .lda_functions import load_mzml_and_make_documents, run_lda, load_dict
from decomposition.models import Beta
from ms2ldaviz.celery_tasks import app


@app.task
def lda_task(exp_id, params):
    exp = Experiment.objects.get(pk=exp_id)
    K = int(params['K'])
    print 'Running lda on experiment_%d (%s), K=%d' % (exp_id, exp.description, K)
    print 'CSV file = %s' % exp.csv_file
    print 'mzML file = %s' % exp.mzml_file

    corpus, metadata, word_mz_range = load_mzml_and_make_documents(exp)
    lda_dict = run_lda(corpus, metadata, word_mz_range, K, n_its=1000)
    load_dict(lda_dict, exp)

    ready, desc = EXPERIMENT_STATUS_CODE[1]
    exp.status = ready
    exp.save()


@app.task
def decomposition_task(exp_id, params):
    exp = Experiment.objects.get(pk=exp_id)
    decompose_from = params['decompose_from']
    print 'Running decomposition on experiment_%d (%s), decompose_from %s' % (exp_id, exp.description,
                                                                              decompose_from)
    print 'CSV file = %s' % exp.csv_file
    print 'mzML file = %s' % exp.mzml_file

    load_mzml_and_make_documents(exp, decompose_from)
    beta = Beta.objects.all()[0]
    documents = Document.objects.filter(experiment=exp)
    decompose(documents, beta)

    ready, desc = EXPERIMENT_STATUS_CODE[1]
    exp.status = ready
    exp.save()
