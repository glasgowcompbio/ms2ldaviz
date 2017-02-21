from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_TYPE
from basicviz.models import UserExperiment
from .forms import CreateExperimentForm
from .tasks import ms2lda_task, decomposition_task


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
            ms2lda: ms2lda_task,
            decomposition: decomposition_task
        }

        # runs the correct pipeline based on the experiment type
        pipeline = pipelines[exp.experiment_type]
        pipeline.delay(exp.id)