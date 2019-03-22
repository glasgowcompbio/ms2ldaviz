import os
import sys
import jsonpickle
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings_simon")
import django
django.setup()
from django.db import transaction


# delete all current instances
from annotation.models import *
from basicviz.models import *

si = SubstituentInstance.objects.all()
ti = TaxaInstance.objects.all()

print "Found {} substituents and {} taxa".format(len(si),len(ti))


si.delete()
ti.delete()

# run the migration at this point!!!

experiment_names = ['massbank_binned_005','gnps_binned_005']

corpus_file_names = ['/home/simon/Dropbox/BioResearch/Meta_clustering/MS2LDA/classyfire/massbank_classyfire_corpora.dict',
                    '/home/simon/Dropbox/BioResearch/Meta_clustering/MS2LDA/classyfire/gnps_classyfire_corpora.dict']


sub_terms = SubstituentTerm.objects.all()
print "Found {} terms".format(len(sub_terms))
term_dict = {t.name:t for t in sub_terms}

# populate the db
for ii,experiment_name in enumerate(experiment_names):
    try:
        mbe = Experiment.objects.get(name = experiment_name)
        print mbe
    except:
        print "No such experiment"
        continue

   

    # load the dict file
    import pickle
    with open(corpus_file_names[ii],'r') as f:
        mb_corpora = pickle.loads(f.read())

    docs = mb_corpora['substituents_corpus'].keys()

    db_docs = Document.objects.filter(experiment = mbe)
    db_doc_dict = {d.name:d for d in db_docs}
    term_list = mb_corpora['substituents_list']
    with transaction.atomic():
        for doc in docs:
            print doc
            binary_vals = mb_corpora['substituents_corpus'][doc]
            this_term_list = [term_list[i] for i,v in enumerate(binary_vals) if v == 1]
            try:
                doc_obj = db_doc_dict[doc]
                for term in this_term_list:
                    term_obj = term_dict[term]
                    SubstituentInstance.objects.get_or_create(document = doc_obj,subterm = term_obj,source='Standard')
            except:
                print "Problem with ",doc

