import json

from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from basicviz.models import Experiment,UserExperiment
from .forms import AnnotationForm, BatchAnnotationForm
from .lda_methods import annotate
from .helpers import deprecated
from .constants import ANNOTATE_DATABASES

from annotation.tasks import predict_substituent_terms
from annotation.models import SubstituentInstance,SubstituentTerm

@login_required(login_url = '/registration/login/')
def index(request):
    ue = UserExperiment.objects.filter(user = request.user)
    experiments = [u.experiment for u in ue]

    tokeep = []
    for e in experiments:
        links = e.multilink_set.all()
        if len(links) == 0:
            tokeep.append(e)
    
    context_dict = {'experiments':tokeep}
    return render(request,'annotation/index.html',context_dict)


def parse_spectrum_string(spectrum_string):
    # Parse the spectrum that has been input
    peaks = []
    tokens = spectrum_string.split()
    mz = None
    intensity = None
    for token in tokens:
        # First check for MONA format
        if ':' in token:
            vals = token.split(':')
            mz = float(vals[0])
            intensity = float(vals[1])
            peaks.append((mz,intensity))
            continue
        else:
            # If not MONA, assume that its just mz, rt pairs in a long list
            if mz is None:
                # Must be a new peak
                mz = float(token)
            elif intensity is None:
                # Already have a mz so this must be an intensity
                intensity = float(token)
                # Store the peak and then forget the mz and intensity
                peaks.append((mz,intensity))
                mz = None
                intensity = None
    return peaks


@login_required(login_url = '/registration/login/')
def start_annotation(request,basicviz_experiment_id):
    # Starts the annotation. User uploads a spectrum    
    experiment = Experiment.objects.get(id = basicviz_experiment_id)
    context_dict = {'experiment':experiment}

    if request.method == 'POST':
        form = AnnotationForm(request.POST)
        if form.is_valid():
            parentmass = form.cleaned_data['parentmass']
            spectrum_string = form.cleaned_data['spectrum']
            peaks = parse_spectrum_string(spectrum_string)

            spectrum = (parentmass,peaks)
            context_dict['spectrum'] = spectrum
            # Do the annotation
            document,motif_theta_overlap,plotdata,taxa_term_probs,sub_term_probs,matches_count =\
                annotate(spectrum,basicviz_experiment_id)
            context_dict['motif_theta_overlap'] = motif_theta_overlap
            context_dict['document'] = document
            context_dict['plotdata'] = json.dumps(plotdata)
            context_dict['taxa_term_probs'] = taxa_term_probs
            context_dict['sub_term_probs'] = sub_term_probs

            # Make the data for the scatter plot
            scatter_data = []
            for m,t,o in motif_theta_overlap:
                text = m.name
                if m.short_annotation:
                    text += ": " + m.short_annotation
                elif m.annotation:
                    text += ": " + m.annotation
                scatter_data.append((t,o,text))

            context_dict['scatter_data'] = json.dumps(scatter_data)

        else:
            context_dict['annotation_form'] = form
    else:
        form = AnnotationForm()
        context_dict['annotation_form'] = form

    return render(request,'annotation/start_annotation.html',context_dict)


# Replaced by batch_query_annotation()
# I think Simon is still using this method in the notebooks..? Otherwise we can delete it.
@deprecated
@csrf_exempt
def query_annotation(request, basicviz_experiment_id):

    response_data = {'status': 'OK'}
    if request.method == "POST":

        form = AnnotationForm(request.POST)
        if form.is_valid():

            parentmass = form.cleaned_data['parentmass']
            spectrum_json = form.cleaned_data['spectrum']
            peaks = json.loads(spectrum_json)

            spectrum = (parentmass, peaks)
            print(spectrum)
            document, motif_theta_overlap, plotdata, taxa_term_probs, sub_term_probs, matches_count = \
                annotate(spectrum, basicviz_experiment_id)

            taxa_terms = []
            for taxa, prob in taxa_term_probs:
                taxa_terms.append((taxa.name, prob))
            response_data['taxa_term_probs'] = taxa_terms

            sub_terms = []
            for sub, prob in sub_term_probs:
                sub_terms.append((sub.name, prob))
            response_data['sub_term_probs'] = sub_terms

            mto = []
            for motif, theta, overlap in motif_theta_overlap:
                mto.append((motif.name, motif.annotation, theta, overlap))
            response_data['motif_theta_overlap'] = mto

            response_data['fragment_match'] = matches_count['fragment']
            response_data['loss_match'] = matches_count['loss']
            response_data['fragment_intensity_match'] = matches_count['fragment_intensity']
            response_data['loss_intensity_match'] = matches_count['loss_intensity']

        else: # form validation failed
            field_errors = [(field.label, field.errors) for field in form]
            response_data['status'] = 'ERROR: %s' % field_errors

    return JsonResponse(response_data)


@csrf_exempt
def batch_query_annotation(request, db_name):

    # check valid db name
    if db_name not in ANNOTATE_DATABASES.keys():
        msg = 'ERROR: annotation can only be performed against %s' % ANNOTATE_DATABASES.keys()
        response_data = {
            'status': msg
        }
        return JsonResponse(response_data)

    basicviz_experiment_id = ANNOTATE_DATABASES[db_name]
    if request.method == "POST":

        form = BatchAnnotationForm(request.POST)
        if form.is_valid():

            spectra_json = form.cleaned_data['spectra']
            spectra = json.loads(spectra_json)
            response_data = {'status': 'OK'}

            for parentmass in spectra:

                peaks = spectra[parentmass]
                parentmass = float(parentmass)
                spectrum = (parentmass, peaks)
                document, motif_theta_overlap, plotdata, taxa_term_probs, sub_term_probs, matches_count = \
                    annotate(spectrum, basicviz_experiment_id)

                response_data[parentmass] = {}

                taxa_terms = []
                for taxa, prob in taxa_term_probs:
                    taxa_terms.append((taxa.name, prob))
                response_data[parentmass]['taxa_term_probs'] = taxa_terms

                sub_terms = []
                for sub, prob in sub_term_probs:
                    sub_terms.append((sub.name, prob))
                response_data[parentmass]['sub_term_probs'] = sub_terms

                mto = []
                for motif, theta, overlap in motif_theta_overlap:
                    mto.append((motif.name, motif.annotation, theta, overlap))
                response_data[parentmass]['motif_theta_overlap'] = mto

                response_data[parentmass]['fragment_match'] = matches_count['fragment']
                response_data[parentmass]['loss_match'] = matches_count['loss']
                response_data[parentmass]['fragment_intensity_match'] = matches_count['fragment_intensity']
                response_data[parentmass]['loss_intensity_match'] = matches_count['loss_intensity']

        else: # form validation failed
            field_errors = [(field.label, field.errors) for field in form]
            response_data = {'status': 'ERROR: %s' % field_errors}

    return JsonResponse(response_data)

def check_user(request, experiment):
    user = request.user
    try:
        ue = UserExperiment.objects.get(experiment=experiment, user=user)
        return ue.permission
    except:
        # try the public experiments
        e = PublicExperiments.objects.filter(experiment = experiment)
        if len(e) > 0:
            return "view"
        else:
            # User can't see this one
            return None


def start_term_prediction(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    if not check_user(request, experiment) == 'edit':
        return HttpResponse("You do not have permission to perform this operation")
    existing_terms = SubstituentInstance.objects.filter(document__experiment = experiment,source = "Predicted")
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['n_existing_terms'] = len(existing_terms)
    return render(request,'annotation/start_term_prediction.html',context_dict)

def delete_predictions(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    if not check_user(request, experiment) == 'edit':
        return HttpResponse("You do not have permission to perform this operation")
    existing_terms = SubstituentInstance.objects.filter(document__experiment = experiment,source = "Predicted")
    existing_terms.delete()
    return redirect('/annotation/start_term_prediction/{}'.format(experiment_id))

@login_required
def term_prediction(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    if not check_user(request, experiment) == 'edit':
        return HttpResponse("You do not have permission to perform this operation")
    predict_substituent_terms.delay(experiment_id)
    return redirect('/basicviz')

def explore_terms(request,experiment_id):
    context_dict = {}
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict['experiment'] = experiment

    terms = SubstituentInstance.objects.filter(document__experiment = experiment)
    term_counts = {}
    for term in terms:
        if term.subterm in term_counts:
            term_counts[term.subterm] += 1
        else:
            term_counts[term.subterm] = 1
    context_dict['term_counts'] = list(zip(term_counts.keys(),term_counts.values()))
    return render(request,'annotation/explore_terms.html',context_dict)


def list_docs_for_term(request,experiment_id,term_id):
    context_dict = {}
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict['experiment'] = experiment
    term = SubstituentTerm.objects.get(id = term_id)
    terms = SubstituentInstance.objects.filter(document__experiment = experiment,subterm = term)
    context_dict['term_instances'] = terms
    context_dict['term'] = term
    return render(request,'annotation/list_docs_for_term.html',context_dict)
    