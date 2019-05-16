import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
import jsonpickle
django.setup()

from chemspipy import ChemSpider

from basicviz.models import *

if __name__ == '__main__':
    cs = ChemSpider('b2VqZPJug1yDvbPgawGdGO59pdBw4eaf')

    exp_name = sys.argv[1]
    e = Experiment.objects.get(name = exp_name)
    print e
    docs = Document.objects.filter(experiment = e)
    for doc in docs:
        md = jsonpickle.decode(doc.metadata)
        ik = md.get('InChIKey',md.get('inchikey',None))
        print ik
        # search in chemspi
        results = cs.search(ik)
        if len(results) > 0:
            m = results[0].mol_2d
            if len(m) > 0:
                doc.mol_string = m
                doc.save()