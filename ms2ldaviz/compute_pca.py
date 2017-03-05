import os
import pickle
import numpy as np
import sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")
import django
django.setup()

from sklearn.decomposition import PCA


import jsonpickle

from basicviz.models import MultiFileExperiment,MultiLink,Experiment,Document,Feature,FeatureInstance,Mass2Motif,Mass2MotifInstance,DocumentMass2Motif,FeatureMass2MotifInstance

from basicviz.views import make_alpha_matrix

if __name__ == '__main__':
    mfname = sys.argv[1]
    mfe = MultiFileExperiment.objects.get(name = mfname)
    links = MultiLink.objects.filter(multifileexperiment = mfe)
    individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']
    motifs = Mass2Motif.objects.filter(experiment = individuals[0]).order_by('name')
    alp_vals = make_alpha_matrix(individuals,normalise = True)
    np_alp = np.array(alp_vals)
    pca = PCA(n_components = 2,whiten = True,copy = True)
    pca.fit(np_alp.T)
    X = pca.transform(np_alp.T)
    # Make the object to pass to the view
    points = []
    lines = []
    for i,individual in enumerate(individuals):
        new_row = [float(X[i,0]),float(X[i,1]),individual.name,'#FF0000']
        points.append(new_row)

    ma = np.abs(pca.components_).max()
    pma = np.abs(X).max()

    scale_factor = 0.5*pma/ma

    for i,motif in enumerate(motifs):
        new_row = [float(pca.components_[0,i]*scale_factor),float(pca.components_[1,i]*scale_factor),motif.name,'#999999']
        lines.append(new_row)
    pca_data = (points,lines)

    encoded = jsonpickle.encode(pca_data)

    mfe.pca = encoded
    mfe.save()
