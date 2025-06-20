import os
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE, EXPERIMENT_TYPE
from basicviz.models import UserExperiment, ExtraUsers, \
    MultiFileExperiment, Experiment


@login_required(login_url='/registration/login/')
def index(request):
    userexperiments = UserExperiment.objects.filter(user=request.user).select_related('experiment').prefetch_related(
        'experiment__analysis_set',
        'experiment__multilink_set',
        'experiment__userexperiment_set',
        'experiment__publicexperiments_set',
    )
    experiments = []
    for ue in userexperiments:
        experiments.append(ue.experiment)

    exclude_individuals = []
    pending_individuals = []
    finished_individuals = []
    show_lda = False
    show_ms1_set = set()  ## dictionary to record whether to show ms1 result option in main page
    for experiment in experiments:

        # Remove those that are multi ones
        links = experiment.multilink_set.all()
        if len(links) > 0:
            exclude_individuals += [l.experiment for l in links]

        # Also exclude those with pending status
        pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0]
        if experiment.status == pending_code:
            exclude_individuals.append(experiment)
            pending_individuals.append(experiment)
        else:  # experiments are already finished
            lda_code, _ = EXPERIMENT_TYPE[0]
            if experiment.experiment_type == lda_code:
                show_lda = True
                finished_individuals.append(experiment)
                if experiment.analysis_set.count() > 0:
                    show_ms1_set.add(experiment.id)

    show_pending = False if len(pending_individuals) == 0 else True
    for e in exclude_individuals:
        del experiments[experiments.index(e)]

    # retrieve the permission of experiments too
    experiments = list(set(experiments))
    included_ids = [e.id for e in experiments]
    ues = UserExperiment.objects.filter(experiment_id__in = included_ids, user=request.user).prefetch_related(
        'experiment__publicexperiments_set',
    ).select_related('experiment', 'user').order_by('-experiment_id')

    exps = []
    permissions = []
    public_status = []
    for ue in ues:
        print(ue.experiment.id, ue.user, ue.experiment, ue.permission)
        exps.append(ue.experiment)
        permissions.append(ue.permission)
        pe = ue.experiment.publicexperiments_set
        is_public = True if pe.count() > 0 else False
        public_status.append(is_public)
    experiments = zip(exps, permissions, public_status)

    # hide the create experiment button for guest user or if disabled by environment variable
    show_create_experiment = True
    if request.user.username.lower() == 'guest':
        show_create_experiment = False

    # Check if job submission is disabled by environment variable
    enable_job_submission = os.environ.get('ENABLE_ORIGINAL_JOB_SUBMISSION', '1')
    if enable_job_submission == '0':
        show_create_experiment = False

    # to display additional links on the basicviz index page
    eu_count = ExtraUsers.objects.filter(user=request.user).count()
    extra_user = True if eu_count > 0 else False

    context_dict = {
        'experiments': experiments,
        'user': request.user,
        'pending_experiments': pending_individuals,
        'show_pending': show_pending,
        'show_lda': show_lda,
        'show_create_experiment': show_create_experiment,
        'show_ms1_set': show_ms1_set,
        'extra_user': extra_user
    }
    return render(request, 'basicviz/basicviz.html', context_dict)


@login_required(login_url='/registration/login/')
def index_mfe(request):
    userexperiments = UserExperiment.objects.filter(user=request.user).select_related('experiment').prefetch_related(
        'experiment__analysis_set',
        'experiment__multilink_set',
        'experiment__userexperiment_set',
        'experiment__publicexperiments_set',
    )
    experiments = []
    for ue in userexperiments:
        experiments.append(ue.experiment)

    exclude_individuals = []
    pending_individuals = []
    finished_individuals = []
    show_lda = False
    show_ms1_set = set()  ## dictionary to record whether to show ms1 result option in main page
    for experiment in experiments:

        # Remove those that are multi ones
        links = experiment.multilink_set.all()
        if len(links) > 0:
            exclude_individuals += [l.experiment for l in links]

        # Also exclude those with pending status
        pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0]
        if experiment.status == pending_code:
            exclude_individuals.append(experiment)
            pending_individuals.append(experiment)
        else:  # experiments are already finished
            lda_code, _ = EXPERIMENT_TYPE[0]
            decomposition_code, _ = EXPERIMENT_TYPE[1]
            if experiment.experiment_type == lda_code:
                show_lda = True
                finished_individuals.append(experiment)
                if experiment.analysis_set.count() > 0:
                    show_ms1_set.add(experiment.id)

    show_pending = False if len(pending_individuals) == 0 else True
    for e in exclude_individuals:
        del experiments[experiments.index(e)]

    # retrieve the permission of experiments too
    experiments = list(set(experiments))
    permissions = []
    public_status = []
    for e in experiments:
        ue = e.userexperiment_set.first()
        print(e, ue.permission)
        permissions.append(ue.permission)
        pe = e.publicexperiments_set
        is_public = True if pe.count() > 0 else False
        public_status.append(is_public)
    experiments = zip(experiments, permissions, public_status)

    # hide the create experiment button for guest user or if disabled by environment variable
    show_create_experiment = True
    if request.user.username.lower() == 'guest':
        show_create_experiment = False

    # Check if job submission is disabled by environment variable
    enable_job_submission = os.environ.get('ENABLE_ORIGINAL_JOB_SUBMISSION', '1')
    if enable_job_submission == '0':
        show_create_experiment = False

    context_dict = {'experiments': experiments}
    context_dict['user'] = request.user
    context_dict['pending_experiments'] = pending_individuals
    context_dict['show_pending'] = show_pending
    context_dict['show_lda'] = show_lda
    context_dict['show_create_experiment'] = show_create_experiment
    context_dict['show_ms1_set'] = show_ms1_set

    # to display additional links on the basicviz index page
    eu_count = ExtraUsers.objects.filter(user=request.user).count()
    extra_user = True if eu_count > 0 else False
    context_dict['extra_user'] = extra_user

    mfe = MultiFileExperiment.objects.all().prefetch_related(
        'multilink_set',
        'multilink_set__experiment',
        'multilink_set__experiment__multilink_set'
    )
    context_dict['mfe'] = mfe

    return render(request, 'basicviz/basicviz_mfe.html', context_dict)
