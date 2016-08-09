import os
import csv
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")


import django

django.setup()

from basicviz.models import Experiment,Document,Mass2Motif,DocumentMass2Motif,FeatureInstance,FeatureMass2MotifInstance,Mass2MotifInstance

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    out_file = sys.argv[3]
    p_thresh = float(sys.argv[2])
    experiment = Experiment.objects.get(name = experiment_name)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    annotated_mass2motifs = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)

    
    with open(out_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['msm_id','m2m_annotation','doc_id','doc_annotation','valid','probability','score'])
        for mass2motif in annotated_mass2motifs:
            print mass2motif
            m2minstances = Mass2MotifInstance.objects.filter(mass2motif = mass2motif)
            m2mfeatures = {}
            for instance in m2minstances:
                m2mfeatures[instance.feature] = instance.probability
            dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = 0.02)
            
            for dm2m in dm2ms:
                score = 0.0
                document = dm2m.document
                feature_instances = FeatureInstance.objects.filter(document = document)
                for instance in feature_instances:
                    if instance.feature in m2mfeatures:
                        fm2m = FeatureMass2MotifInstance.objects.filter(featureinstance = instance,mass2motif = mass2motif)
                        if fm2m:
                            score += m2mfeatures[instance.feature] * fm2m[0].probability
                # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
                doc_name = '"' + dm2m.document.display_name + '"'
                annotation = '"' + mass2motif.annotation + '"'
                writer.writerow([mass2motif.id,mass2motif.name,mass2motif.annotation.encode('utf8'),dm2m.document.id,doc_name.encode('utf8'),dm2m.validated,dm2m.probability,score])

