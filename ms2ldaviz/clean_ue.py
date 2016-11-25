import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import Experiment, UserExperiment, User

if __name__ == '__main__':
	users = User.objects.all()
	experiments = Experiment.objects.all()

	for user in users:
		for experiment in experiments:
			ue = UserExperiment.objects.filter(user = user,experiment = experiment)
			if len(ue) > 1:
				highest = 'view'
				for u in ue:
					if u.permission == 'edit':
						highest = 'edit'
				for i,u in enumerate(ue):
					if i == 0:
						u.permission = highest
						u.save()
					else:
						u.delete()
