import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

import jsonpickle

from basicviz.models import Experiment
from ms1analysis.models import Sample,DocSampleIntensity


if __name__ == '__main__':
	metfamily = Experiment.objects.get(name = 'metfamily')
	docs = metfamily.document_set.all()
	print("Found {} documents".format(len(docs)))
	doc0 = docs[0]
	md = jsonpickle.decode(doc0.metadata)
	sample_dict = {}
	for key in md['intensities']:
		sample_dict[key],status = Sample.objects.get_or_create(name = key,experiment = metfamily)

	for document in docs:
		for samp,intensity in jsonpickle.decode(document.metadata)['intensities'].items():
			ds,status = DocSampleIntensity.objects.get_or_create(document = document,sample = sample_dict[samp])
			ds.intensity = intensity
			ds.save() 