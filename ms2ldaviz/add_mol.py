import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
import jsonpickle
django.setup()

from chemspipy import ChemSpider
from django.conf import settings

from basicviz.models import *

if __name__ == '__main__':
    cs = ChemSpider(settings.CHEMSPIDER_APIKEY)

    exp_name = sys.argv[1]
    e = Experiment.objects.get(name = exp_name)
    print(e)
    docs = Document.objects.filter(experiment = e).filter(mol_string__isnull=True)
    for doc in docs:
        md = jsonpickle.decode(doc.metadata)
        ik = md.get('InChIKey',md.get('inchikey',None))
        if not ik:
            next
        print(ik)
        try:
            mol = cs.convert(ik,'InChIKey','mol')
            if mol:
                doc.mol_string = mol
                doc.save()
        except Exception as e:
            print('Failed mol fetch of ' + ik + ' : ')
            print(e)
