import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django
django.setup()
from django.db import transaction
import sys
import json

from basicviz.models import Experiment, Mass2Motif, Document, FeatureInstance, MagmaSub, FeatureInstance2Sub

if __name__ == '__main__':

    experiment = sys.argv[1]
    magma_json = sys.argv[2]

    e = Experiment.objects.get(name = experiment)
    with open(magma_json) as f:
        data = json.load(f)

    with transaction.atomic():
        e.has_magma_annotation = True
        e.save()

        i = 0
        feature_map = {}
        for d in data:
            # update the mol string of a document
            name = d['name']
            print('%d/%d %s' % (i+1, len(data), name))
            document = Document.objects.get(experiment=e, name=name)
            document.mol_string = d['mol']
            document.save()

            # look for existing feature instances in this document
            feature_instances = FeatureInstance.objects.filter(document=document)
            for f in feature_instances:
                feature_map[f.feature.name] = f

            # create a new substructure object and link to feature instances
            for magma_annot in d['features']:
                name = magma_annot['name']
                sub_type = magma_annot['type']
                feature = feature_map[name]
                for m in magma_annot['matches']:
                    smiles = m['smiles']
                    fragatoms = m['fragatoms']
                    mz = m['mz']
                    mol = m['mol'] if 'mol' in m else None
                    sub, _ = MagmaSub.objects.get_or_create(smiles=smiles, mol_string=mol)
                    f2sub = FeatureInstance2Sub.objects.get_or_create(
                        feature=feature, sub=sub, fragatoms=fragatoms,
                        mz=mz, sub_type=sub_type
                    )
            i+=1
