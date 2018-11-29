# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render,redirect,HttpResponse
from django.views.decorators.csrf import csrf_exempt

import json

from motifdb.models import *
from basicviz.models import *   

from motifdb.forms import MatchMotifDBForm
from motifdb.tasks import start_motif_matching_task

# Create your views here.
def index(request):
    context_dict = {}

    motif_sets = MDBMotifSet.objects.all()
    context_dict['motif_sets'] = motif_sets
    return render(request, 'motifdb/index.html', context_dict)

def motif_set(request,motif_set_id):
    ms = MDBMotifSet.objects.get(id = motif_set_id)
    context_dict = {}
    context_dict['motif_set'] = ms
    motifs = MDBMotif.objects.filter(motif_set = ms)
    context_dict['motifs'] = motifs
    return render(request,'motifdb/motif_set.html',context_dict)

def motif(request,motif_id):
    motif = MDBMotif.objects.get(id = motif_id)
    context_dict = {}
    context_dict['motif'] = motif
    features = motif.mass2motifinstance_set.all().order_by('-probability')
    context_dict['features'] = features

    maxp = 1.0

    jcamp = '##TITLE={} (fragments)'.format(motif.name)
    jcamp+='\n##XUNITS= M/Z\n##YUNITS= RELATIVE ABUNDANCE\n'
    # jcamp+= '##BASE PEAK INTENSITY= {}\n'.format(maxp)
    jcamp+='##PEAK TABLE= (XY..XY)\n'
    # jcamp+='0,1\n'
    jcamp_loss = '##TITLE={} (losses)'.format(motif.name)
    jcamp_loss+='\n##XUNITS= M/Z\n##YUNITS= RELATIVE ABUNDANCE\n'
    # jcamp_loss += '##BASE PEAK INTENSITY= {}\n'.format(maxp)
    jcamp_loss+='##PEAK TABLE= (XY..XY)\n'
    # jcamp_loss+='0,1\n'
    for f in features:
        if f.feature.name.startswith('fragment'):
            mz = float(f.feature.name.split('_')[1])
            jcamp += "{}, {}\n".format(mz,f.probability)
        else:
            mz = float(f.feature.name.split('_')[1])
            jcamp_loss += "{}, {}\n".format(mz,f.probability)
            
    jcamp += '##END=\n'
    jcamp_loss += '##END=\n'
    context_dict['jcamp'] = jcamp
    context_dict['jcamp_loss'] = jcamp_loss
    return render(request,'motifdb/motif.html',context_dict)

def start_motif_matching(request, experiment_id):
    context_dict = {}
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict['experiment'] = experiment
    if request.method == 'POST':
        match_motif_form = MatchMotifDBForm(request.POST)
        if match_motif_form.is_valid():
            motif_set = match_motif_form.cleaned_data['motif_set']
            motif_set_id = motif_set.id
            minimum_score_to_save = float(match_motif_form.cleaned_data['min_score_to_save'])
            start_motif_matching_task.delay(experiment_id,motif_set_id,minimum_score_to_save)
            # match_motifs.delay(experiment.id, base_experiment_id, min_score_to_save=minimum_score_to_save)
            # match_motifs_set.delay(experiment.id,base_experiment.id, min_score_to_save = minimum_score_to_save)
            return redirect('/basicviz/manage_motif_matches/{}'.format(experiment_id))
    else:
        match_motif_form = MatchMotifDBForm()
    context_dict['match_motif_form'] = match_motif_form
    return render(request, 'motifdb/start_match_motifs.html', context_dict)

def list_motifsets(request):
    motif_sets = MDBMotifSet.objects.all()
    output = {m.name:m.id for m in motif_sets}
    return HttpResponse(json.dumps(output), content_type='application/json') 

def get_motifset(request,motifset_id):
    motifset = MDBMotifSet.objects.get(id = motifset_id)
    motifs = MDBMotif.objects.filter(motif_set = motifset)
    output_motifs = {}
    for motif in motifs:
        fis = Mass2MotifInstance.objects.filter(mass2motif = motif)
        output_motifs[motif.name] = {}
        for fi in fis:
            output_motifs[motif.name][fi.feature.name] = fi.probability
    return HttpResponse(json.dumps(output_motifs), content_type='application/json')

@csrf_exempt
def get_motifset_post(request):
    motifset_id = int(request.POST['motifset_id'])
    motifset = MDBMotifSet.objects.get(id = motifset_id)
    motifs = MDBMotif.objects.filter(motif_set = motifset)
    output_motifs = {}
    for motif in motifs:
        fis = Mass2MotifInstance.objects.filter(mass2motif = motif)
        output_motifs[motif.name] = {}
        for fi in fis:
            output_motifs[motif.name][fi.feature.name] = fi.probability
    return HttpResponse(json.dumps(output_motifs), content_type='application/json')

def get_motifset_metadata(request,motifset_id):
    motifset = MDBMotifSet.objects.get(id = motifset_id)
    motifs = MDBMotif.objects.filter(motif_set = motifset)
    output_motifs = {}
    for motif in motifs:
        md = jsonpickle.decode(motif.metadata)
        output_motifs[motif.name] = md
    return HttpResponse(json.dumps(output_motifs), content_type='application/json')
