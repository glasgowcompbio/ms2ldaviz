## This script is used to solve the problem if we want to set different threshold values for probability and overlap
## Original settings use *default_doc_m2m_score* and *doc_m2m_threshold*
## Here, we use this script to delete them, and use *doc_m2m_prob_threshold* and *doc_m2m_overlap_threshold* to replace
import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

from basicviz.models import Experiment,User,UserExperiment,SystemOptions
from options.views import get_option


exp_id_set = set()
for opt in SystemOptions.objects.all():
	exp_id_set.add(opt.experiment_id)

## check if something wrong with fetching doc_m2m_threshold data
for exp_id in exp_id_set:
	if exp_id:
		experiment = Experiment.objects.get(id = exp_id)
		doc_m2m_threshold = get_option('doc_m2m_threshold', experiment=experiment)
		if not doc_m2m_threshold:
			print("!!!get_option for expreriment {} failed".format(exp_id))
			break

'''
Updating Rules:
for one experiment:
	if not doc_m2m_threshold setting:
		1. if no default_doc_m2m_score
			=> DO NOTHING
		2. if default_doc_m2m_score is 'probability'
			=> DO NOTHING
		3. if default_doc_m2m_score is 'overlap_score'
			=> probability threshold to 0.0
			=> overlap threshold to original gloabl doc_m2m_threshold
		4. if default_doc_m2m_score is 'both'
			=> overlap threshold to original gloabl doc_m2m_threshold
	if have doc_m2m_threshold setting:
		get this value to be doc_m2m_threshold
		5. if no default_doc_m2m_score
			=> probability threshold to doc_m2m_threshold
		6. if default_doc_m2m_score is 'probability'
			=> probability threshold to doc_m2m_threshold
		7. if default_doc_m2m_score is 'overlap_score'
			=> probability threshold to 0.0
			=> overlap threshold to doc_m2m_threshold
		8. if default_doc_m2m_score is 'both'
			=> probability threshold to doc_m2m_threshold
			=> overlap threshold to doc_m2m_threshold

for global setting:
	=> probability threshold to 0.05
	=> overlap threshold to 0.0
'''

## update Experiment-specific options
global_doc_m2m_threshold = float(SystemOptions.objects.filter(key='doc_m2m_threshold', experiment__isnull = True)[0].value)
for exp_id in exp_id_set:
	if exp_id:
		experiment = Experiment.objects.get(id = exp_id)
		score_options = SystemOptions.objects.filter(key='default_doc_m2m_score', experiment=experiment)
		threshold_options = SystemOptions.objects.filter(key='doc_m2m_threshold', experiment=experiment)
		if len(threshold_options) == 0:
			if len(score_options) == 0:
				continue
			else:
				default_score = score_options[0].value
				if default_score == "probability":
					continue
				elif default_score == "overlap_score":
					SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = 0.0, experiment = experiment)
					SystemOptions.objects.create(key = "doc_m2m_overlap_threshold", value = global_doc_m2m_threshold, experiment = experiment)
				elif default_score == "both":
					SystemOptions.objects.create(key = "doc_m2m_overlap_threshold", value = global_doc_m2m_threshold, experiment = experiment)
				
		else:
			doc_m2m_threshold = float(threshold_options[0].value)
			if len(score_options) == 0:
				SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = doc_m2m_threshold, experiment = experiment)
			else:
				default_score = score_options[0].value
				if default_score == "probability":
					SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = doc_m2m_threshold, experiment = experiment)
				elif default_score == "overlap_score":
					SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = 0.0, experiment = experiment)
					SystemOptions.objects.create(key = "doc_m2m_overlap_threshold", value = doc_m2m_threshold, experiment = experiment)
				elif default_score == "both":
					SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = doc_m2m_threshold, experiment = experiment)
					SystemOptions.objects.create(key = "doc_m2m_overlap_threshold", value = doc_m2m_threshold, experiment = experiment)
				

## update Global options:
SystemOptions.objects.create(key = "doc_m2m_prob_threshold", value = 0.05, experiment = None)
SystemOptions.objects.create(key = "doc_m2m_overlap_threshold", value = 0.0, experiment = None)


### TO DELETE
## The delete operations have been commented out
## If everything works fine, manually deleting seems to be more safer
# SystemOptions.objects.filter(key = "doc_m2m_threshold").delete()
# SystemOptions.objects.filter(key = "default_doc_m2m_score").delete()

