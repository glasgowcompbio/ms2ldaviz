import os

from django.conf import settings
from django.contrib.auth.decorators import user_passes_test
from django.http import StreamingHttpResponse
from django.shortcuts import render

from basicviz.models import JobLog, Experiment, UserExperiment


@user_passes_test(lambda u: u.is_staff)
def list_log(request):
    logs = JobLog.objects.all().select_related(
        'experiment',
        'user'
    )
    context_dict = {}
    context_dict['logs'] = logs
    return render(request,'basicviz/log_list.html',context_dict)


@user_passes_test(lambda u: u.is_staff)
def show_log_file(request, experiment_id):
    try:
        media_folder = settings.MEDIA_ROOT
        log_folder = os.path.join(media_folder, 'logs')
        log_file = os.path.join(log_folder, 'experiment_' + experiment_id + '.log')
        content = open(log_file, 'r').read()
    except IOError:
        content = 'No file is found.'
    response = StreamingHttpResponse(content)
    response['Content-Type'] = 'text/plain; charset=utf8'
    return response


@user_passes_test(lambda u: u.is_staff)
def list_all_experiments(request):
    experiments = Experiment.objects.all().prefetch_related(
        'userexperiment_set',
        'joblog_set'
    )
    results = []
    for experiment in experiments:
        user_experiments = UserExperiment.objects.filter(experiment=experiment)
        joblog = JobLog.objects.filter(experiment=experiment).first()
        pe = user_experiments.experiment.publicexperiments_set
        is_public = True if pe.count() > 0 else False
        row = [experiment, user_experiments, joblog, is_public]
        results.append(row)

    context_dict = {}
    context_dict['results'] = results
    return render(request, 'basicviz/list_all_experiments.html', context_dict)