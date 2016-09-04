import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

from django.contrib.auth.models import User


import django
django.setup()


from basicviz.models import Experiment,User,UserExperiment

if __name__ == '__main__':
	username = sys.argv[1]
	experiment_name = sys.argv[2]
	experiment = Experiment.objects.get(name = experiment_name)
	user = User.objects.get(username = username)
	UserExperiment.objects.create(user = user,experiment = experiment)


