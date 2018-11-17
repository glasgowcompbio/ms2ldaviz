# populating the motifdb web app from motifdb
import os
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")



from django.db import transaction
import django
django.setup()

import glob,jsonpickle

from motifdb.models import *
from basicviz.models import *
if __name__ == '__main__':
    dbpath = '/Users/simon/git/motifdb/motifs'
    motif_sets = glob.glob(dbpath+os.sep + '*')

    sys.path.append('/Users/simon/git/motifdb/code/utilities')
    
    from motifdb_loader import load_db

    fs = BVFeatureSet.objects.get(name = 'binned_005')
    with transaction.atomic():
        for motif_set in motif_sets:
            name = motif_set.split(os.sep)[-1]
            mbs,_ = MDBMotifSet.objects.get_or_create(name = name,featureset = fs)
            motifs,metadata,_ = load_db([name],dbpath)
            for motif,spec in motifs.items():
                print motif
                print metadata[motif]
                
                m,_ = MDBMotif.objects.get_or_create(motif_set = mbs,name = motif)
                md = metadata[motif]
                m.metadata = jsonpickle.encode(md)
                # m.annotation = metadata[motif]['annotation']
                # m.comment = metadata[motif]['comment']
                # m.short_annotation = metadata[motif]['short_annotation']
                m.save()
                
                for feature,probability in spec.items():
                    f,_ = Feature.objects.get_or_create(name = feature,featureset = fs)
                    a,_ = Mass2MotifInstance.objects.get_or_create(feature = f,mass2motif = m,probability = probability)
                    a.save()
            



        