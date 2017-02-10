from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from basicviz.models import UserExperiment, ExtraUsers, \
    MultiFileExperiment, MultiLink


@login_required(login_url='/registration/login/')
def index(request):
    userexperiments = UserExperiment.objects.filter(user=request.user)
    experiments = []
    for ue in userexperiments:
        experiments.append(ue.experiment)

    # Remove those that are multi ones
    exclude_individuals = []

    for experiment in experiments:
        links = MultiLink.objects.filter(experiment=experiment)
        if len(links) > 0:
            exclude_individuals += [l.experiment for l in links]

    for e in exclude_individuals:
        del experiments[experiments.index(e)]

    experiments = list(set(experiments))

    permissions = []
    for e in experiments:
        ue = UserExperiment.objects.get(user = request.user,experiment = e)
        print experiment,ue.permission
        permissions.append(ue.permission)

    experiments = zip(experiments,permissions)

    # experiments = Experiment.objects.all()
    context_dict = {'experiments': experiments}
    context_dict['user'] = request.user

    eu = ExtraUsers.objects.filter(user=request.user)

    mfe = MultiFileExperiment.objects.all()

    if len(eu) > 0:
        extra_user = True
    else:
        extra_user = False
    context_dict['extra_user'] = extra_user
    context_dict['mfe'] = mfe
    return render(request, 'basicviz/basicviz.html', context_dict)