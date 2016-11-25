import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import Experiment, UserExperiment, User

if __name__ == '__main__':
	username = sys.argv[1]
	experimentname = sys.argv[2]
	permission = sys.argv[3]

	experiment = Experiment.objects.get(name = experimentname)
	user = User.objects.filter(username = username)[0]

	ues = UserExperiment.objects.filter(user = user,experiment = experiment)
	if len(ues) == 0:
		UserExperiment.objects.create(user = user,experiment = experiment,permission = permission)
	else:
		for i,ue in enumerate(ues):
			if i==0:
				ue.permission = permission
				ue.save()
			else:
				ue.delete()

	
