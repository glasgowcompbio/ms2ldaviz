from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.models import UserExperiment,JobLog, Document, Experiment
from .forms import AnalysisForm
from .models import Sample, DocSampleIntensity, Analysis, AnalysisResult, AnalysisResultPlage
from .models import DecompositionAnalysis, DecompositionAnalysisResult, DecompositionAnalysisResultPlage
import numpy as np
from scipy.stats import ttest_ind
from ms2ldaviz.celery_tasks import app
from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Mass2Motif, DocumentMass2Motif
from basicviz.views import get_docm2m
from decomposition.decomposition_functions import get_docglobalm2m
from decomposition.models import DocumentGlobalMass2Motif, Decomposition

# Create your views here.
@login_required(login_url='/registration/login/')
def create_ms1analysis(request, experiment_id):
    experiment = Experiment.objects.filter(id=experiment_id)[0]
    context_dict = {}
    context_dict['experiment'] = experiment
    samples = Sample.objects.filter(experiment_id=experiment_id)
    sample_choices = [sample.name for sample in samples]
    sample_choices = sorted(sample_choices)

    if request.method == 'POST':
        analysis_form = AnalysisForm(request.POST)
        group1_choices = request.POST.getlist('group1')
        group2_choices = request.POST.getlist('group2')

        if analysis_form.is_valid() and group1_choices and group2_choices:
            pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0]
            new_analysis = Analysis.objects.get_or_create(name=analysis_form.cleaned_data['name'],
                                                          description=analysis_form.cleaned_data['description'],
                                                          experiment=experiment,
                                                          group1=group1_choices,
                                                          group2=group2_choices,
                                                          status=pending_code
                                                          )[0]

            context_dict['analysis'] = new_analysis
            params = {
                'group1': group1_choices,
                'group2': group2_choices,
                'experiment_id': experiment_id
            }
            process_ms1_analysis.delay(new_analysis.id, params)
            # process_ms1_analysis(new_analysis.id, params)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['analysis_form'] = analysis_form
            context_dict['sample_choices'] = sample_choices

    else:
        analysis_form = AnalysisForm()
        context_dict['analysis_form'] = analysis_form
        context_dict['sample_choices'] = sample_choices

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

    # do PLAGE here
    mass2motifs = Mass2Motif.objects.filter(experiment_id=experiment_id)
    groups = group1 + group2
    samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in groups]
    ## set up a dictionary to cache documents' intensities
    document_intensities_dict = {}
    for mass2motif in mass2motifs:
        docm2ms = get_docm2m(mass2motif)
        # docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif)
        sub_mat = []
        for docm2m in docm2ms:
            if docm2m.document in document_intensities_dict:
                temp_list = document_intensities_dict[docm2m.document]
                sub_mat.append(temp_list)
            else:
                temp_list = []
                for sample in samples:
                    ## missing intensity will be set to 0.0
                    try:
                        intensity = DocSampleIntensity.objects.filter(document=docm2m.document, sample=sample)[0].intensity
                    except:
                        intensity = 0.0
                    temp_list.append(intensity)
                ## documents with all missing values should still be omitted
                if np.sum(temp_list) > 0.0:
                    document_intensities_dict[docm2m.document] = temp_list
                    sub_mat.append(temp_list)

        sub_mat = np.array(sub_mat)

        ## is it correct to set t_val to be zero here???
        if len(sub_mat) == 0:
            true_t_val = 0
            plage_p_val = None
        else:
            u, s, v = np.linalg.svd(sub_mat)
            v0_group1 = v[0][:len(group1)]
            v0_group2 = v[0][len(group1):]
            t_val = np.abs(np.mean(v0_group1) - np.mean(v0_group2))
            t_val /= np.sqrt(np.var(v0_group1)/(1.0*len(group1)) + np.var(v0_group2)/(1.0*len(group2)))
            true_t_val = t_val
            if true_t_val == 0:
                plage_p_val = None
            else:
                count = 0
                iterations = 10000
                for i in range(iterations):
                    v0_permutation = np.random.permutation(v[0])
                    v0_group1 = v0_permutation[:len(group1)]
                    v0_group2 = v0_permutation[len(group1):]
                    t_val = np.abs(np.mean(v0_group1) - np.mean(v0_group2))
                    t_val /= np.sqrt(np.var(v0_group1) / (1.0 * len(group1)) + np.var(v0_group2) / (1.0 * len(group2)))
                    if t_val >= true_t_val:
                        count += 1
                plage_p_val = count * 1.0 / iterations
        AnalysisResultPlage.objects.get_or_create(analysis=new_analysis, mass2motif=mass2motif,plage_t_value=true_t_val, plage_p_value=plage_p_val)

    ## do fold change and pValue here
    for document in documents:
        group1_intensities = get_group_intensities(group1_samples, document, use_logarithm)
        group2_intensities = get_group_intensities(group2_samples, document, use_logarithm)
        if not group1_intensities or not group2_intensities:
            fold = 1
            pValue = None
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
            if not (pValue >= 0 and pValue <= 1):
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


@login_required(login_url='/registration/login/')
## replicate from *create_ms1analysis* function for LDA
## in decomposition we need to store *decomposition* information, instead of *experiment*
## and *experiment* can be got through decomposition.experiment_id
def create_ms1analysis_decomposition(request, decomposition_id):
    decomposition = Decomposition.objects.filter(id=decomposition_id)[0]
    experiment = Experiment.objects.get(id=decomposition.experiment_id)
    context_dict = {}
    context_dict['decomposition'] = decomposition

    samples = Sample.objects.filter(experiment=experiment)
    sample_choices = [sample.name for sample in samples]
    sample_choices = sorted(sample_choices)

    if request.method == 'POST':
        analysis_form = AnalysisForm(request.POST)
        group1_choices = request.POST.getlist('group1')
        group2_choices = request.POST.getlist('group2')

        if analysis_form.is_valid() and group1_choices and group2_choices:
            pending_code, pending_msg = EXPERIMENT_STATUS_CODE[0]
            new_analysis = DecompositionAnalysis.objects.get_or_create(name=analysis_form.cleaned_data['name'],
                                                          description=analysis_form.cleaned_data['description'],
                                                          decomposition=decomposition,
                                                          group1=group1_choices,
                                                          group2=group2_choices,
                                                          status=pending_code
                                                          )[0]

            context_dict['analysis'] = new_analysis
            params = {
                'group1': group1_choices,
                'group2': group2_choices,
                'decomposition_id': decomposition_id
            }
            process_ms1_analysis_decomposition.delay(new_analysis.id, params)
            # process_ms1_analysis_decomposition(new_analysis.id, params)
            return HttpResponseRedirect(reverse('index'))

        else:
            context_dict['analysis_form'] = analysis_form
            context_dict['sample_choices'] = sample_choices

    else:
        analysis_form = AnalysisForm()
        context_dict['analysis_form'] = analysis_form
        context_dict['sample_choices'] = sample_choices

    return render(request, 'ms1analysis/add_ms1_analysis.html', context_dict)

@app.task
## replicate from *process_ms1_analysis* function for LDA
## need to notice Decomposition used an extra layer *GlobalMotif* to keep things clean,
## and relationship between documents and motifs are stored in *DocumentGlobalMass2Motif*
def process_ms1_analysis_decomposition(new_analysis_id, params):
    new_analysis = DecompositionAnalysis.objects.get(pk=new_analysis_id)
    group1 = params['group1']
    group2 = params['group2']
    decomposition_id = params['decomposition_id']
    decomposition = Decomposition.objects.get(id=decomposition_id)
    experiment_id = Decomposition.objects.get(id=decomposition_id).experiment_id
    use_logarithm = new_analysis.use_logarithm

    group1_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group1]
    group2_samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in group2]

    documents = Document.objects.filter(experiment_id=experiment_id)

    # do PLAGE here
    ## to keep the name consistent with *LDA experiment*, still use *mass2motif*, *mass2motifs*, *docm2ms*...
    ## but need to notice here motif refer to *GlobalMotif*, not *Mass2Motif*
    docm2ms = DocumentGlobalMass2Motif.objects.filter(decomposition_id=decomposition_id)
    mass2motifs = set([docm2m.mass2motif for docm2m in docm2ms])

    # mass2motifs = Mass2Motif.objects.filter(experiment_id=experiment_id)
    groups = group1 + group2
    samples = [Sample.objects.filter(name=sample_name, experiment_id=experiment_id)[0] for sample_name in groups]
    ## set up a dictionary to cache documents' intensities
    document_intensities_dict = {}
    for mass2motif in mass2motifs:
        docm2ms = get_docglobalm2m(mass2motif, decomposition)
        sub_mat = []
        for docm2m in docm2ms:
            if docm2m.document in document_intensities_dict:
                temp_list = document_intensities_dict[docm2m.document]
                sub_mat.append(temp_list)
            else:
                temp_list = []
                for sample in samples:
                    ## missing intensity will be set to 0.0
                    try:
                        intensity = DocSampleIntensity.objects.filter(document=docm2m.document, sample=sample)[0].intensity
                    except:
                        intensity = 0.0
                    temp_list.append(intensity)
                ## documents with all missing values should still be omitted
                if np.sum(temp_list) > 0.0:
                    document_intensities_dict[docm2m.document] = temp_list
                    sub_mat.append(temp_list)

        sub_mat = np.array(sub_mat)

        if len(sub_mat) == 0:
            true_t_val = 0
            plage_p_val = None
        else:
            u, s, v = np.linalg.svd(sub_mat)
            v0_group1 = v[0][:len(group1)]
            v0_group2 = v[0][len(group1):]
            t_val = np.abs(np.mean(v0_group1) - np.mean(v0_group2))
            t_val /= np.sqrt(np.var(v0_group1)/(1.0*len(group1)) + np.var(v0_group2)/(1.0*len(group2)))
            true_t_val = t_val
            if true_t_val == 0:
                plage_p_val = None
            else:
                count = 0
                iterations = 10000
                for i in range(iterations):
                    v0_permutation = np.random.permutation(v[0])
                    v0_group1 = v0_permutation[:len(group1)]
                    v0_group2 = v0_permutation[len(group1):]
                    t_val = np.abs(np.mean(v0_group1) - np.mean(v0_group2))
                    t_val /= np.sqrt(np.var(v0_group1) / (1.0 * len(group1)) + np.var(v0_group2) / (1.0 * len(group2)))
                    if t_val >= true_t_val:
                        count += 1
                plage_p_val = count * 1.0 / iterations
        DecompositionAnalysisResultPlage.objects.get_or_create(analysis=new_analysis, globalmotif=mass2motif,plage_t_value=true_t_val, plage_p_value=plage_p_val)

    ## do fold change and pValue here
    ## fold change and pValue are basically similar to LDA experiment
    ## since the way we get Sample and DocSampleIntensity is identical to LDA
    ## the only difference here is the *new_analysis* object if from DecompositionAnalysis,
    ## so the result for *fold change* and *pValue* should be stored in DecompositionAnalysisResult
    for document in documents:
        group1_intensities = get_group_intensities(group1_samples, document, use_logarithm)
        group2_intensities = get_group_intensities(group2_samples, document, use_logarithm)
        if not group1_intensities or not group2_intensities:
            fold = 1
            pValue = None
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
            if not (pValue >= 0 and pValue <= 1):
                pValue = None
        # add_analysis_result(new_analysis, document, fold, pValue)
        DecompositionAnalysisResult.objects.get_or_create(analysis=new_analysis, document=document, foldChange=fold, pValue=pValue)

    ready, _ = EXPERIMENT_STATUS_CODE[1]
    new_analysis.status = ready
    new_analysis.save()