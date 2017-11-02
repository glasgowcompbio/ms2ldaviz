import os,sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import *
from django.contrib.auth.models import User

if __name__ == '__main__':
	username = sys.argv[1]
	try:
		user = User.objects.get(username = username)
	except:
		print "No such user! {}".format(username)
		sys.exit(0)

	experiment_list = ['Beer6_POS_IPA_MS1_comparisons',
	    'Urine37_POS_StandardLDA_300Mass2Motifs_MS1peaklist_MS1duplicatefilter',
	    'Beer3_POS_Decomposition_MassBankGNPScombinedset_MS1peaklist_MS1duplicatefilter',
	    'massbank_binned_005',
	    'gnps_binned_005',
	    'Beer3_POS_StandardLDA_300Mass2Motifs_MS1peaklist_MS1duplicatefilter']

	experiments = []
	for ename in experiment_list:
		try:
			e = Experiment.objects.get(name = ename)
			experiments.append(e)
		except:
			print "No such experiment: {}".format(ename)

	for e in experiments:
		ue = UserExperiment.objects.filter(user = user,experiment = e)
		if len(ue) == 0:
			if e.name.startswith('Beer6'):
				UserExperiment.objects.create(user = user,experiment = e,permission = 'edit')
			else:	
				UserExperiment.objects.create(user = user,experiment = e,permission = 'view')
