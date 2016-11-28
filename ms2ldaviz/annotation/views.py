import json
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from basicviz.models import Experiment,UserExperiment
from annotation.forms import AnnotationForm
from lda_methods import annotate


@login_required(login_url = '/basicviz/login/')
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
            if mz == None:
                # Must be a new peak
                mz = float(token)
            elif intensity == None:
                # Already have a mz so this must be an intensity
                intensity = float(token)
                # Store the peak and then forget the mz and intensity
                peaks.append((mz,intensity))
                mz = None
                intensity = None
    return peaks




@login_required(login_url = '/basicviz/login/')
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
            document,motif_theta_overlap,plotdata = annotate(spectrum,basicviz_experiment_id)
            context_dict['motif_theta_overlap'] = motif_theta_overlap
            context_dict['document'] = document
            context_dict['plotdata'] = json.dumps(plotdata)


        else:
            context_dict['annotation_form'] = form
    else:
        form = AnnotationForm()
        context_dict['annotation_form'] = form

    return render(request,'annotation/start_annotation.html',context_dict)