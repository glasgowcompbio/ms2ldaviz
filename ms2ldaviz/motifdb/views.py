# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from motifdb.models import *

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
    features = motif.mdbmotifinstance_set.all().order_by('-probability')
    context_dict['features'] = features

    maxp = 1.0

    jcamp = '##TITLE={} (fragments)'.format(motif.name)
    jcamp+='\n##XUNITS= M/Z\n##YUNITS= RELATIVE ABUNDANCE\n'
    jcamp+= '##BASE PEAK INTENSITY = {}\n'.format(maxp)
    jcamp+='##PEAK TABLE= (XY..XY)\n'

    jcamp_loss = '##TITLE={} (losses)'.format(motif.name)
    jcamp_loss+='\n##XUNITS= M/Z\n##YUNITS= RELATIVE ABUNDANCE\n'
    jcamp_loss += '##BASE PEAK INTENSITY = {}\n'.format(maxp)
    jcamp_loss+='##PEAK TABLE= (XY..XY)\n'

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