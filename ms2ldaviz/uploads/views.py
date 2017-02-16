from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from basicviz.models import Experiment, MultiFileExperiment, UserExperiment
from django.shortcuts import render

from .forms import CreateExperimentForm
from basicviz.constants import EXPERIMENT_STATUS_CODE

@login_required(login_url='/registration/login/')
def create_experiment(request):

    context_dict = {}
    if request.method == 'POST':
        experiment_form = CreateExperimentForm(request.POST)
        if experiment_form.is_valid():
            new_experiment = experiment_form.save(commit=False)
            pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0] # pending
            new_experiment.status = pending_code
            new_experiment.save()
            user = request.user
            UserExperiment.objects.create(user=user, experiment=new_experiment, permission='edit')
            return HttpResponseRedirect(reverse('index'))
        else:
            context_dict['experiment_form'] = experiment_form
    else:
        experiment_form = CreateExperimentForm()
        context_dict['experiment_form'] = experiment_form

    return render(request, 'uploads/add_experiment.html', context_dict)