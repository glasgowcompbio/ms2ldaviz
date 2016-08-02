import os
import sys
import jsonpickle
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")
import django
django.setup()

from basicviz.models import Experiment,Document

if __name__ == '__main__':
	from chemspipy import ChemSpider
	cs = ChemSpider('b07b7eb2-0ba7-40db-abc3-2a77a7544a3d')

	experiment_name = sys.argv[1]
	experiment = Experiment.objects.get(name = experiment_name)
	documents = Document.objects.filter(experiment = experiment)

	for document in documents:
		print document
		md = jsonpickle.decode(document.metadata)
		csid = md.get('csid',-1)
		if csid > -1:
			c = cs.get_compound(csid)
			md['common_name'] = c.common_name
			document.metadata = jsonpickle.encode(md)
			document.save()
			print c.common_name

