import os

from django.contrib.auth.decorators import login_required,user_passes_test
from django.shortcuts import render
from basicviz.models import JobLog
from django.http import StreamingHttpResponse
from django.conf import settings


@user_passes_test(lambda u: u.is_staff)
def list_log(request):
    logs = JobLog.objects.all()
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
