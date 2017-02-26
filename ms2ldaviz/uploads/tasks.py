from __future__ import absolute_import, unicode_literals

from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Document, Experiment
from decomposition.decomposition_functions import load_mzml_and_make_documents,decompose
from decomposition.models import Beta
from ms2ldaviz.celery_tasks import app


@app.task
def lda_task(exp_id, params):

    exp = Experiment.objects.get(pk=exp_id)
    print 'Running MS2LDA on experiment_%d (%s)' % (exp_id, exp.description)
    print 'CSV file = %s' % exp.csv_file
    print 'mzML file = %s' % exp.mzml_file

    # TODO: do stuff here

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
    documents = Document.objects.filter(experiment = exp)
    decompose(documents,beta)

    ready, desc = EXPERIMENT_STATUS_CODE[1]
    exp.status = ready
    exp.save()
