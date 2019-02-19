import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

experiment_id = 601

from basicviz.models import *

e = Experiment.objects.get(id = experiment_id)
mm = MotifMatch.objects.filter(frommotif__experiment = e)
filtered_mm = filter(lambda x: x.tomotif.experiment == None,mm)
print len(filtered_mm)
filtered_mm.sort(key = lambda x: x.score,reverse = True)
from basicviz.views.views_lda_single import get_docm2m
unique_docs = set()
count = 1
for fmm in filtered_mm:
   print count,fmm.tomotif,fmm.score
   dm = get_docm2m(fmm.frommotif)
   for d in dm:
      unique_docs.add(d.document)
   print len(dm),len(unique_docs)
   count +=1 
print len(Document.objects.filter(experiment = e))
print len(Mass2Motif.objects.filter(experiment = e))
