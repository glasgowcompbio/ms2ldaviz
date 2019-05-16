import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()

from basicviz.models import *
from motifdb.models import *

if __name__ == '__main__':
    motifset_id = int(sys.argv[1])
    experiment_id = int(sys.argv[2])



    experiment = Experiment.objects.get(id = experiment_id)
    original_motifs = Mass2Motif.objects.filter(experiment = experiment)

    print experiment

    m_dict = {m.name:m for m in original_motifs}

    motif_set = MDBMotifSet.objects.get(id = motifset_id)
    mdb_motifs = MDBMotif.objects.filter(motif_set = motif_set)

    print motif_set

    for m in mdb_motifs:
        old_name = '_'.join(m.name.split('_')[1:]).split('.')[0]
        old_motif = m_dict[old_name]
        print m,old_motif
        m.linkmotif = old_motif
        m.save()