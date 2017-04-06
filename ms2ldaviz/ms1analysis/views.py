from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.models import UserExperiment,JobLog, Document, Experiment
from .forms import AnalysisForm
from .models import Sample, DocSampleIntensity, AnalysisResult
import numpy as np
from scipy.stats import ttest_ind


# Create your views here.
@login_required(login_url='/registration/login/')
def create_ms1analysis(request, experiment_id):
    experiment = Experiment.objects.filter(id=experiment_id)[0]
    context_dict = {}
    context_dict['experiment_id'] = experiment_id
    samples = Sample.objects.filter(experiment_id=experiment_id)
    sample_choices = [(sample.name, sample.name) for sample in samples]

    if request.method == 'POST':
        analysis_form = AnalysisForm(sample_choices, request.POST)
        if analysis_form.is_valid():
            new_analysis = analysis_form.save(commit=False)
            new_analysis.experiment = experiment
            new_analysis.save()

            context_dict['analysis'] = new_analysis
            process_ms1_analysis(new_analysis, analysis_form.cleaned_data, experiment_id)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['analysis_form'] = analysis_form

    else:
        analysis_form = AnalysisForm(sample_choices)
        context_dict['analysis_form'] = analysis_form

    return render(request, 'ms1analysis/add_ms1_analysis.html', context_dict)

def process_ms1_analysis(new_analysis, cleaned_data, experiment_id):
    group1 = cleaned_data['group1']
    group2 = cleaned_data['group2']
    group1_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group1]
    group2_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group2]

    documents = Document.objects.filter(experiment_id=experiment_id)

    for document in documents:
        group1_intensities = []
        group2_intensities = []
        for sample in group1_samples:
            group1_intensities.append(DocSampleIntensity.objects.filter(sample=sample, document=document)[0].intensity)
        for sample in group2_samples:
            group2_intensities.append(DocSampleIntensity.objects.filter(sample=sample, document=document)[0].intensity)
        fold = np.mean(group1_intensities) / np.mean(group2_intensities)
        pValue = ttest_ind(group1_intensities, group2_intensities, equal_var = False)[1]
        add_analysis_result(new_analysis, document, fold, pValue)
        # print fold, pValue

def add_analysis_result(new_analysis, document, fold, pValue):
    result = AnalysisResult.objects.get_or_create(analysis=new_analysis, document=document, foldChange=fold, pValue=pValue)


    # params = {
    #     'decompose_from': decompose_name,
    #     'K': cleaned_data['K'],
    #     'n_its': cleaned_data['n_its'],
    # }
    # do_analysis.delay(new_analysis.id, params)
    #

    # pending, desc = EXPERIMENT_STATUS_CODE[0]
    # if exp.status == pending:
    #
    #     # select the right pipeline
    #     lda, desc = EXPERIMENT_TYPE[0]
    #     decomposition, desc = EXPERIMENT_TYPE[1]
    #     pipelines = {
    #         lda: lda_task,
    #         decomposition: decomposition_task
    #     }
    #
    #     # runs the correct pipeline based on the experiment type
    #     decompose_from = cleaned_data['decompose_from']
    #     decompose_name = decompose_from.name if decompose_from is not None else None
    #     params = {
    #         'decompose_from': decompose_name,
    #         'K': cleaned_data['K'],
    #         'n_its': cleaned_data['n_its'],
    #     }
    #     pipeline = pipelines[exp.experiment_type]
    #
    #
    #     pipeline.delay(exp.id, params)