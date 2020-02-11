import json
from ms2ldaviz.celery_tasks import app

from .decomposition_functions import make_documents,api_decomposition
from decomposition.models import FeatureSet,MotifSet,APIBatchResult
@app.task
def api_batch_task(spectra,featureset_id,motifset_id,result_id):
	featureset = FeatureSet.objects.get(id = featureset_id)
	motifset = MotifSet.objects.get(id = motifset_id)
	doc_dict = make_documents(spectra,featureset)
	results = api_decomposition(doc_dict,motifset)
	batch_results = APIBatchResult.objects.get(id = result_id)
	batch_results.results = json.dumps(results)
	batch_results.save()

