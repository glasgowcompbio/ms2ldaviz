import json

from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from basicviz.models import Experiment, MultiFileExperiment, UserExperiment
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Experiment, UserExperiment
from .forms import CreateExperimentForm

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
            return HttpResponseRedirect(reverse('index'))
        else:
            context_dict['experiment_form'] = experiment_form
    else:
        experiment_form = CreateExperimentForm()
        context_dict['experiment_form'] = experiment_form

    return render(request, 'uploads/add_experiment.html', context_dict)