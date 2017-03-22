from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_TYPE
from basicviz.models import UserExperiment,JobLog
from .forms import CreateExperimentForm
from .tasks import lda_task, decomposition_task


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
            new_experiment.ms2_file = request.FILES.get('ms2_file', None)
            new_experiment.save()

            user = request.user
            UserExperiment.objects.create(user=user, experiment=new_experiment, permission='edit')

            JobLog.objects.create(user = user, experiment = new_experiment, tasktype = 'Uploaded and ' + new_experiment.experiment_type)


            process_experiment(new_experiment, experiment_form.cleaned_data)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['experiment_form'] = experiment_form
    else:
        experiment_form = CreateExperimentForm()
        context_dict['experiment_form'] = experiment_form

    return render(request, 'uploads/add_experiment.html', context_dict)


def process_experiment(exp, cleaned_data):

    pending, desc = EXPERIMENT_STATUS_CODE[0]
    if exp.status == pending:

        # select the right pipeline
        lda, desc = EXPERIMENT_TYPE[0]
        decomposition, desc = EXPERIMENT_TYPE[1]
        pipelines = {
            lda: lda_task,
            decomposition: decomposition_task
        }

        # runs the correct pipeline based on the experiment type
        decompose_from = cleaned_data['decompose_from']
        decompose_name = decompose_from.name if decompose_from is not None else None
        params = {
            'decompose_from': decompose_name,
            'K': cleaned_data['K'],
            'n_its': cleaned_data['n_its'],
        }
        pipeline = pipelines[exp.experiment_type]


        pipeline.delay(exp.id, params)