import sys
import jsonpickle
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

from basicviz.models import Experiment,Document

import django
django.setup()

# This script copies document metadata from one experiment to another
# Command line arguments should be from_experiment_name to_experiment_name

if __name__ == '__main__':
	from_experiment_name = sys.argv[1]
	to_experiment_name = sys.argv[2]
	try: 
		from_experiment = Experiment.objects.get(name = from_experiment_name)
	except: 
		print("From experiment not found")
		sys.exit(0)

	try:
		to_experiment = Experiment.objects.get(name = to_experiment_name)
	except:
		print("To experiment not found")
		sys.exit(0)

	print("Found experiments")
	from_documents = Document.objects.filter(experiment = from_experiment)
	print("Extracted {} documents from from_experiment".format(len(from_documents)))
	n_found = 0
	for document in from_documents:
		to_document = Document.objects.filter(experiment = to_experiment,name = document.name)
		if len(to_document) == 1:
			print("Found match for {}".format(document.name))
			n_found += 1
			to_document = to_document[0]
			to_metadata = {}
			from_metadata = jsonpickle.decode(document.metadata)
			for key,value in from_metadata.items():
				to_metadata[key] = value
			to_document.metadata = jsonpickle.encode(to_metadata)
			to_document.save()
	print("Found {} matches".format(n_found))

