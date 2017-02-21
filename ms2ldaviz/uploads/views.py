import json

from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from basicviz.models import Experiment, MultiFileExperiment, UserExperiment
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_TYPE
from basicviz.models import Experiment, UserExperiment, Document
from .forms import CreateExperimentForm

from decomposition.decomposition_functions import load_mzml_and_make_documents,decompose
from decomposition.models import Beta

@login_required(login_url='/registration/login/')
def create_experiment(request):

    context_dict = {}
    if request.method == 'POST':
        experiment_form = CreateExperimentForm(request.POST, request.FILES)
        if experiment_form.is_valid():

            new_experiment = experiment_form.save(commit=False)
            pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0] # pending
            new_experiment.status = pending_code
            new_experiment.csv_file = request.FILES.get('csv_file', None)
            new_experiment.mzml_file = request.FILES.get('mzml_file', None)
            new_experiment.save()

            user = request.user
            UserExperiment.objects.create(user=user, experiment=new_experiment, permission='edit')

            process_experiment(new_experiment)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['experiment_form'] = experiment_form
    else:
        experiment_form = CreateExperimentForm()
        context_dict['experiment_form'] = experiment_form

    return render(request, 'uploads/add_experiment.html', context_dict)


def process_experiment(exp):

    pending, desc = EXPERIMENT_STATUS_CODE[0]
    ready, desc = EXPERIMENT_STATUS_CODE[1]
    if exp.status == pending:

        # select the right pipeline
        ms2lda, desc = EXPERIMENT_TYPE[0]
        decomposition, desc = EXPERIMENT_TYPE[1]
        pipelines = {
            ms2lda: lda_pipeline,
            decomposition: decomposition_pipeline
        }
        pipeline = pipelines[exp.experiment_type] # this is a function

        # runs the correct pipeline based on the experiment type
        pipeline(exp)

        # finished
        exp.status = ready
        exp.save()


def lda_pipeline(exp):
    print 'Running LDA pipeline'
    print exp.csv_file, exp.mzml_file


def decomposition_pipeline(exp):
    print 'Running decomposition pipeline'
    print exp.csv_file, exp.mzml_file
    load_mzml_and_make_documents(exp)
    beta = Beta.objects.all()[0]
    documents = Document.objects.filter(experiment = exp)
    decompose(documents,beta)
    exp.status = "1"
    exp.save()
