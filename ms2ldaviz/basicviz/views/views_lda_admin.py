from django.contrib.auth.decorators import login_required,user_passes_test
from django.shortcuts import render
from basicviz.models import JobLog

@user_passes_test(lambda u: u.is_staff)
def list_log(request):
	logs = JobLog.objects.all()
	context_dict = {}
	context_dict['logs'] = logs
	return render(request,'basicviz/log_list.html',context_dict)