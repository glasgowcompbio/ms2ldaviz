import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()


from basicviz.models import Experiment,User,UserExperiment

if __name__ == '__main__':
	username = sys.argv[1]
	if len(sys.argv) > 2:
		email = sys.argv[2]
		pw = sys.argv[3]
	user = User.objects.filter(username = username)
	if len(user) == 0:
		user = User.objects.create_user(username, email, pw)
	else:
		user = user[0]
	experiments = Experiment.objects.all()

	for experiment in experiments:
		UserExperiment.objects.create(user = user,experiment = experiment)

