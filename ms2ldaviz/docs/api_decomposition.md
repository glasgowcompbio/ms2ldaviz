# MS2LDA API Decomposition
# Usage Description
# Simon Rogers, March 2017

The API batch decomposition features allows a user to send a bunch of spectra to the server to be decomposed onto a particular <motifset>.
The spectra are passed as the arguments to a POST request to the following url:
http://ms2lda.org/decomposition/api/batch_decompose/

The argument should be a dictionary with the following two `<key,value>` pairs:

Key: ‘motifset’ Value: name of the motifset to decompose onto, e.g. ‘massbank_motifset’

Key: ‘spectra’ Value: the spectral information, pickled into a string (using e.g. json.dumps)

The spectra value should be a list, with one item per spectra. The item should be a tuple with three elements: (string: doc_name, float: parentmass, list: peaks)

Peaks is a list of tuples, each representing a peak in the form (float: mz, float: intensity)

For example in python, using the requests package

~~~~{.python}

import requests
import json

spectrum = ('spec_name',188.0818,[(53.0384,331117.7),
(57.0447,798106.4),
(65.0386,633125.7),
(77.0385,5916789.799999999),
(81.0334,27067.0),
(85.0396,740633.6)])

spectra = [spectrum] # or add more to the list

args = {'spectra': json.dumps(spectra), 'motifset': 'massbank_motifset'}

url = 'http://ms2lda.org/decomposition/api/batch_decompose/'

r = requests.post(url,args)

~~~~

because this is computationally intensive, the decomposition is run as a celery task. Therefore the post request doesn’t return the results. Instead it returns some summary including the id of the results entry. To get the results (in json), do the following:


~~~~{.python}

result_id = r.json()['result_id']
url2 = 'http://ms2lda.org/decomposition/api/batch_results/{}/'.format(result_id)
r2 = request.get(url2)
print r2.json()

~~~~

If `r2.json()` has a ‘status’ field, it means the job is still running / waiting. If not, you get a dictionary back with the document names as keys and a list as the value. Each list element has the form:
`(string:globalmofitname,string:originalmotifname,float:theta,float:overlap_score)`

Todo:

 - Add Joe’s nice logging to the task
 - Add the decomps into the JobLog table
 - Put the annotations into the output json
