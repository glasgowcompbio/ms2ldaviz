import requests
import json
from .helpers import deprecated


@deprecated
def annotate(parentmass, spectrum, experiment_id):
    url = 'http://ms2lda.org/annotation/query/%d/' % experiment_id

    params = {
        'parentmass': parentmass,
        'spectrum': json.dumps(spectrum)
    }
    response = requests.post(url, data=params)
    try:
        data = json.loads(response.text)
        return data
    except:
        print response.text


def batch_annotate(spectra, db_name):

    # url = 'http://ms2lda.org/annotation/batch_query/%s/' % db_name
    url = 'http://localhost:8001/annotation/batch_query/%s/' % db_name

    params = {
        'spectra': json.dumps(spectra)
    }
    response = requests.post(url, data=params)
    try:
        data = json.loads(response.text)
        return data
    except:
        print response.text


def print_annotation(data, no_features=None, theta_threshold=0.01):
    print 'Taxa terms'
    for taxa_term, prob in data['taxa_term_probs']:
        if prob > 0.1:
            print taxa_term, prob

    print '\nSubtituent terms'
    for sub_term, prob in data['sub_term_probs']:
        print sub_term, prob

    print '\nMotifs'
    for motif, annotation, theta, overlap in data['motif_theta_overlap']:
        if theta > theta_threshold:
            print motif, annotation, theta, overlap

    if no_features is not None:
        print '\nNo. of fragments matched: %d/%d' % (data['fragment_match'], no_features)
        print '\nNo. of losses matched: %d/%d' % (data['loss_match'], no_features)
    else:
        print '\nNo. of fragments matched: %d' % data['fragment_match'], no_features
        print '\nNo. of losses matched: %d' % data['loss_match'], no_features