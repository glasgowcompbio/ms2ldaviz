import os
from django.contrib.auth.decorators import login_required, user_passes_test
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_TYPE, EXPERIMENT_MS2_FORMAT
from basicviz.models import ExtraUsers, UserExperiment, JobLog, Experiment
from basicviz.views import list_all_experiments
from .forms import CreateExperimentForm, UploadExperimentForm, UploadGensimExperimentForm
from .tasks import lda_task, decomposition_task, upload_task, upload_gensim_task


@login_required(login_url='/registration/login/')
def create_experiment(request):
    # Check if job submission is disabled by environment variable
    enable_job_submission = os.environ.get('ENABLE_ORIGINAL_JOB_SUBMISSION', '1')
    if enable_job_submission == '0':
        # If job submission is disabled, redirect to index page
        return HttpResponseRedirect(reverse('index'))

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

    # to display additional links on the basicviz index page
    eu = ExtraUsers.objects.filter(user=request.user)
    if len(eu) > 0:
        extra_user = True
    else:
        extra_user = False
    context_dict['extra_user'] = extra_user

    return render(request, 'uploads/add_experiment.html', context_dict)


@login_required(login_url='/registration/login/')
def upload_experiment(request):
    # Check if job submission is disabled by environment variable
    enable_job_submission = os.environ.get('ENABLE_ORIGINAL_JOB_SUBMISSION', '1')
    if enable_job_submission == '0':
        # If job submission is disabled, redirect to index page
        return HttpResponseRedirect(reverse('index'))

    context_dict = {}
    if request.method == 'POST':
        experiment_form = UploadExperimentForm(request.POST, request.FILES)
        if experiment_form.is_valid():
            new_experiment = experiment_form.save(commit=False)
            new_experiment.status = EXPERIMENT_STATUS_CODE[0][0]
            new_experiment.experiment_ms2_format = EXPERIMENT_MS2_FORMAT[3][0]
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
        experiment_form = UploadExperimentForm()
        context_dict['experiment_form'] = experiment_form

    # to display additional links on the basicviz index page
    eu = ExtraUsers.objects.filter(user=request.user)
    if len(eu) > 0:
        extra_user = True
    else:
        extra_user = False
    context_dict['extra_user'] = extra_user

    return render(request, 'uploads/upload_experiment.html', context_dict)


@login_required(login_url='/registration/login/')
def upload_gensim_experiment(request):
    # Check if job submission is disabled by environment variable
    enable_job_submission = os.environ.get('ENABLE_ORIGINAL_JOB_SUBMISSION', '1')
    if enable_job_submission == '0':
        # If job submission is disabled, redirect to index page
        return HttpResponseRedirect(reverse('index'))

    context_dict = {}
    if request.method == 'POST':
        experiment_form = UploadGensimExperimentForm(request.POST, request.FILES)
        if experiment_form.is_valid():
            new_experiment = experiment_form.save(commit=False)
            new_experiment.status = EXPERIMENT_STATUS_CODE[0][0]
            new_experiment.experiment_ms2_format = EXPERIMENT_MS2_FORMAT[4][0]
            new_experiment.ms2_file = request.FILES.get('ms2_file', None)
            new_experiment.csv_file = request.FILES.get('csv_file', None)
            new_experiment.save()

            user = request.user
            UserExperiment.objects.create(user=user, experiment=new_experiment, permission='edit')

            JobLog.objects.create(user = user, experiment = new_experiment, tasktype = 'Uploaded and ' + new_experiment.experiment_type)
            process_experiment(new_experiment, experiment_form.cleaned_data)
            return HttpResponseRedirect(reverse('index'))
        else:
            context_dict['experiment_form'] = experiment_form
    else:
        experiment_form = UploadGensimExperimentForm()
        context_dict['experiment_form'] = experiment_form

    return render(request, 'uploads/upload_gensim_experiment.html', context_dict)


def process_experiment(exp, cleaned_data):

    pending, desc = EXPERIMENT_STATUS_CODE[0]
    if exp.status == pending:

        # runs the correct task based on the experiment type and ms2 format
        params = {}
        task = None
        if exp.experiment_type == EXPERIMENT_TYPE[1][0]: # run decomposition
            params['decompose_from'] = cleaned_data['decompose_from'].name if 'decompose_from' in cleaned_data else None
            task = decomposition_task
        elif exp.experiment_type == EXPERIMENT_TYPE[0][0]: # run upload of lda results
            if exp.experiment_ms2_format == EXPERIMENT_MS2_FORMAT[3][0]:
                params['filename'] = exp.ms2_file.path
                params['featureset'] = exp.featureset.name
                task = upload_task
            elif exp.experiment_ms2_format == EXPERIMENT_MS2_FORMAT[4][0]:
                params['corpus_filename'] = exp.ms2_file.path
                params['gensim_filename'] = exp.csv_file.path
                params['featureset'] = exp.featureset.name
                task = upload_gensim_task
            else: # run lda inference
                params['K'] = cleaned_data['K'] if 'K' in cleaned_data else None
                params['n_its'] = cleaned_data['n_its'] if 'n_its' in cleaned_data else None
                task = lda_task

        task.delay(exp.id, params)


@user_passes_test(lambda u: u.is_staff)
def rerun_experiment(request, experiment_id):
    exp = Experiment.objects.get(id = experiment_id)
    print(exp)
    params = {}
    if exp.experiment_ms2_format == EXPERIMENT_MS2_FORMAT[3][0]:
        params['filename'] = exp.ms2_file.path
        params['featureset'] = exp.featureset.name
        task = upload_task
    elif exp.experiment_ms2_format == EXPERIMENT_MS2_FORMAT[4][0]:
        params['corpus_filename'] = exp.ms2_file.path
        params['gensim_filename'] = exp.csv_file.path
        params['featureset'] = exp.featureset.name
        task = upload_gensim_task
    else:  # run lda inference
        params['K'] = exp.K
        params['n_its'] = exp.n_its
        task = lda_task

    task.delay(exp.id, params)
    return list_all_experiments(request)
