
from ms2ldaviz.celery_tasks import app

from django.db import transaction

import numpy as np
import pandas as pd
from keras.models import load_model

from basicviz.models import Document, FeatureInstance
from annotation.models import SubstituentTerm, SubstituentInstance



@app.task
def predict_substituent_terms(experiment_id):

    documents = Document.objects.filter(experiment__id=experiment_id)
    doc_names = []
    doc_dict = {}
    for doc in documents:
        doc_names.append(doc.name)
        doc_dict[doc.name] = doc


    MAX_MASS = 1000
    BIN_SIZE = 1
    intensities = pd.DataFrame(0.0, index=doc_names, columns=range((MAX_MASS // BIN_SIZE)), dtype=float)

    for doc in documents:
        feature_instances = FeatureInstance.objects.filter(document__id=doc.id)
        # get the mass and convert to nearest int bin e.g. mass 2.9 will be put in bin 1 (0-999)
        for feature_instance in feature_instances:
            feature_type = feature_instance.feature.name.split('_')[0]
            if feature_type == 'fragment':
                mass = float(feature_instance.feature.name.split("_")[1])
                if mass < MAX_MASS:
                    mass_bin = (int(mass) // BIN_SIZE) - 1
                    intensity = feature_instance.intensity

                    # Populate the dataframe using each document's name to place data in the correct row.
                    intensities.at[doc.name, mass_bin] = intensity

    np_matrix = intensities.values
    np_index = intensities.index
    x_train_spectra = np.log(np_matrix + 1)

    model = load_model('/home/classyfire/saved_substituents_classifier_model.h5')
    predicted = model.predict(x_train_spectra)

    print len(predicted),len(documents)

    # turn the probabilities into 0 and 1
    predicted_substituents = np.zeros(predicted.shape)
    predicted_substituents[predicted > 0.8] = 1

    legend_all = pd.read_csv("/home/classyfire/filtered_top_substituents_legend.txt", names=["name"])
    legends = legend_all["name"].values.tolist()
    
    sub_terms = SubstituentTerm.objects.all()
    sub_term_dict = {s.name: s for s in sub_terms}

    print predicted
    store_thresh = 0.7
    with transaction.atomic():
        for doc_idx,doc_preds in enumerate(predicted):
            for term_idx,probability in enumerate(doc_preds):
                if probability >= store_thresh:
                    term = legends[term_idx]
                    doc_name = np_index[doc_idx]
                    document = doc_dict[doc_name]
                    if term in sub_term_dict:
                        subterm = sub_term_dict[term]
                    else:
                        subterm = SubstituentTerm.objects.get_or_create(name=term)[0]
                        sub_term_dict[term] = subterm
                    subintance = SubstituentInstance.objects.get_or_create(subterm=subterm, document=document)[0]
                    subintance.probability = probability
                    subintance.source = "Predicted"
                    subintance.save()


    # for index, value in enumerate(predicted_substituents):
    #     if value:
    #         print value
    #         doc_name = np_index[index]
    #         term = legends[index]
    #         probability = predicted[index]
    #         document = Document.objects.get(name=doc_name, experiment_id=experiment_id)
    #         subterm = SubstituentTerm.objects.get_or_create(name=term)[0]
    #         subintance = SubstituentInstance.objects.get_or_create(subterm=subterm, document=document)
    #         subintance.probability = probability
    #         subintance.save()
