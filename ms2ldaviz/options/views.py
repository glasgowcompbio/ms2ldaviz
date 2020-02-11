from django.shortcuts import render

from basicviz.models import Experiment, MultiFileExperiment, SystemOptions
from options.constants import AVAILABLE_OPTIONS
from options.forms import SystemOptionsForm


def get_option(key, experiment=None):
    # Retrieves an option, looking for an experiment specific one if it exists
    print("Looking for option ",key)
    if experiment:
        options = SystemOptions.objects.filter(key=key, experiment=experiment)
        if len(options) == 0:
            options = SystemOptions.objects.filter(key=key, experiment__isnull = True)
    else:
        options = SystemOptions.objects.filter(key=key, experiment__isnull = True)
    if len(options) > 0:
        return options[0].value 
    else:
        return None


def view_experiment_options(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    global_options = SystemOptions.objects.filter(experiment=None)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['global_options'] = global_options
    specific_options = SystemOptions.objects.filter(experiment=experiment)
    context_dict['specific_options'] = specific_options

    return render(request, 'options/view_experiment_options.html', context_dict)


def add_experiment_option(request, experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['available'] = AVAILABLE_OPTIONS
    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            new_option = option_form.save(commit=False)
            new_option.experiment = experiment
            new_option.save()

            return view_experiment_options(request, experiment.id)

        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm()
        context_dict['option_form'] = option_form

    return render(request, 'options/add_experiment_option.html', context_dict)


def delete_experiment_option(request, option_id):
    option = SystemOptions.objects.get(id=option_id)
    experiment = option.experiment
    option.delete()
    return view_experiment_options(request, experiment.id)


def edit_experiment_option(request, option_id):
    option = SystemOptions.objects.get(id=option_id)
    experiment = option.experiment

    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST, instance=option)
        if option_form.is_valid():
            option_form.save()
            return view_experiment_options(request, experiment.id)
        else:
            context_dict = {}
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm(instance=option)
        context_dict = {}
        context_dict['experiment'] = experiment
        context_dict['option_form'] = option_form
        context_dict['option'] = option
        context_dict['available'] = AVAILABLE_OPTIONS
    return render(request, 'options/edit_experiment_option.html', context_dict)


def view_mf_experiment_options(request, mfe_id):
    mfe = MultiFileExperiment.objects.get(id=mfe_id)
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    first_experiment = individuals[0]

    global_options = SystemOptions.objects.filter(experiment=None)
    specific_options = SystemOptions.objects.filter(experiment=first_experiment)

    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['global_options'] = global_options
    context_dict['specific_options'] = specific_options

    return render(request, 'options/view_mf_experiment_options.html', context_dict)


def add_mf_experiment_option(request, mfe_id):
    mfe = MultiFileExperiment.objects.get(id=mfe_id)
    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['available'] = AVAILABLE_OPTIONS
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]

    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            for experiment in individuals:
                key = option_form.cleaned_data['key']
                value = option_form.cleaned_data['value']
                new_option = SystemOptions.objects.get_or_create(experiment=experiment, key=key)[0]
                new_option.value = value
                new_option.save()

            return view_mf_experiment_options(request, mfe.id)

        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm()
        context_dict['option_form'] = option_form

    return render(request, 'options/add_mf_experiment_option.html', context_dict)


def delete_mf_experiment_option(request, option_id):
    option = SystemOptions.objects.get(id=option_id)
    experiment = option.experiment
    link = experiment.multilink_set.all()
    mfe = link[0].multifileexperiment
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    key = option.key
    for experiment in individuals:
        option = SystemOptions.objects.get(experiment=experiment, key=key)
        option.delete()

    return view_mf_experiment_options(request, mfe.id)


def edit_mf_experiment_option(request, option_id):
    option = SystemOptions.objects.get(id=option_id)
    experiment = option.experiment
    link = experiment.multilink_set.all()
    mfe = link[0].multifileexperiment
    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['option'] = option
    context_dict['available'] = AVAILABLE_OPTIONS
    links = mfe.multilink_set.all().order_by('experiment__name')
    individuals = [l.experiment for l in links]
    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            key = option_form.cleaned_data['key']
            value = option_form.cleaned_data['value']
            for experiment in individuals:
                new_option = SystemOptions.objects.get_or_create(experiment=experiment, key=key)[0]
                new_option.value = value
                new_option.save()
            return view_mf_experiment_options(request, mfe.id)
        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm(instance=option)
        context_dict['option_form'] = option_form

    return render(request, 'options/edit_mf_experiment_option.html', context_dict)

