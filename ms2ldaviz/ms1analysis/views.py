from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.models import UserExperiment,JobLog, Document, Experiment
from .forms import AnalysisForm
from .models import Sample, DocSampleIntensity, Analysis, AnalysisResult
import numpy as np
from scipy.stats import ttest_ind
from ms2ldaviz.celery_tasks import app
from basicviz.constants import EXPERIMENT_STATUS_CODE


# Create your views here.
@login_required(login_url='/registration/login/')
def create_ms1analysis(request, experiment_id):
    experiment = Experiment.objects.filter(id=experiment_id)[0]
    context_dict = {}
    context_dict['experiment'] = experiment
    samples = Sample.objects.filter(experiment_id=experiment_id)
    sample_choices = [(sample.name, sample.name) for sample in samples]
    sample_choices = sorted(sample_choices, key = lambda tup: tup[0])

    if request.method == 'POST':
        analysis_form = AnalysisForm(sample_choices, request.POST)
        if analysis_form.is_valid():
            new_analysis = analysis_form.save(commit=False)
            pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0]
            new_analysis.status = pending_code
            new_analysis.experiment = experiment
            new_analysis.save()

            context_dict['analysis'] = new_analysis
            params = {
                'group1': analysis_form.cleaned_data['group1'],
                'group2': analysis_form.cleaned_data['group2'],
                'experiment_id': experiment_id
            }
            process_ms1_analysis.delay(new_analysis.id, params)
            # process_ms1_analysis(new_analysis.id, params)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['analysis_form'] = analysis_form

    else:
        analysis_form = AnalysisForm(sample_choices)
        context_dict['analysis_form'] = analysis_form

    return render(request, 'ms1analysis/add_ms1_analysis.html', context_dict)

@app.task
def process_ms1_analysis(new_analysis_id, params):
    new_analysis = Analysis.objects.get(pk=new_analysis_id)
    group1 = params['group1']
    group2 = params['group2']
    experiment_id = params['experiment_id']
    use_logarithm = new_analysis.use_logarithm

    group1_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group1]
    group2_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group2]

    documents = Document.objects.filter(experiment_id=experiment_id)

    for document in documents:
        group1_intensities = get_group_intensities(group1_samples, document, use_logarithm)
        group2_intensities = get_group_intensities(group2_samples, document, use_logarithm)
        if not group1_intensities or not group2_intensities:
            fold = 1
        else:
            ## intensities between 0 and 1 are very rare
            ## if this happens, it will influence other documents' colouring
            ## so label fold to be 1 here (with white colour) to overcome that
            if np.mean(group1_intensities) <= 1 or np.mean(group2_intensities) <= 1:
                fold = 1
            else:
                fold = np.mean(group1_intensities) / np.mean(group2_intensities)
            if len(group1_intensities) > 1 and len(group2_intensities) > 1:
                try:
                    pValue = ttest_ind(group1_intensities, group2_intensities, equal_var = False)[1]
                except:
                    pValue = None
            else:
                pValue = None
            if not pValue >= 0 and not pValue <= 1:
                pValue = None
        # add_analysis_result(new_analysis, document, fold, pValue)
        AnalysisResult.objects.get_or_create(analysis=new_analysis, document=document, foldChange=fold, pValue=pValue)

    ready, _ = EXPERIMENT_STATUS_CODE[1]
    new_analysis.status = ready
    new_analysis.save()


def get_group_intensities(group_samples, document, use_logarithm='N'):
    group_intensities = []
    for sample in group_samples:
        query_res = DocSampleIntensity.objects.filter(sample=sample, document=document)
        if query_res:
            if use_logarithm == 'Y':
                group_intensities.append(np.log(query_res[0].intensity))
            elif use_logarithm == 'N':
                group_intensities.append(query_res[0].intensity)
    return group_intensities
