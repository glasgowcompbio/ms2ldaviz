import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


from basicviz.models import Experiment,User,UserExperiment

if __name__ == '__main__':
	username = sys.argv[1]
	user = User.objects.get(username = username)
	experiments = Experiment.objects.all()
	for experiment in experiments:
		ue = UserExperiment.objects.get_or_create(user = user,experiment = experiment)[0]
		ue.permission = 'edit'
		ue.save()
