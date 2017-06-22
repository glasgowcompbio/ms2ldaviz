from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from basicviz.constants import EXPERIMENT_STATUS_CODE
from basicviz.models import Experiment
from decomposition.models import Decomposition
from .forms import AnalysisForm
from .models import DecompositionAnalysis
from .models import Sample, Analysis
from .tasks import process_ms1_analysis, process_ms1_analysis_decomposition


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

