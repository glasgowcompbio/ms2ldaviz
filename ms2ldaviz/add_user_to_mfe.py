import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")



import django
django.setup()

from basicviz.models import *
from django.contrib.auth.models import User


if __name__ == '__main__':
	mfename = sys.argv[1]
	username = sys.argv[2]
	user = User.objects.get(username = username)
	mfe = MultiFileExperiment.objects.get(name = mfename)

	links= MultiLink.objects.filter(multifileexperiment = mfe)
	print "Found {} experiments".format(len(links))
	experiments = [l.experiment for l in links]
	for e in experiments:
		ue,_ = UserExperiment.objects.get_or_create(user = user,experiment = e)
		ue.permission = 'edit'
		ue.save()
