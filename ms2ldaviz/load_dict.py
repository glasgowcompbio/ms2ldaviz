import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")

import django
django.setup()

from load_dict_functions import *


if __name__ == '__main__':

    filename = sys.argv[1]
    verbose = False
    if 'verbose' in sys.argv:
        verbose = True
    with open(filename,'r') as f:
        lda_dict = pickle.load(f)
    experiment_name = filename.split('/')[-1].split('.')[0]
    current_e = Experiment.objects.filter(name = experiment_name)
    if len(current_e) > 0:
        print "Experiment of this name already exists, exiting"
        sys.exit(0)

    experiment = Experiment(name = experiment_name)
    experiment.status = 'loading'
    experiment.save()

    load_dict(lda_dict,experiment,verbose)
