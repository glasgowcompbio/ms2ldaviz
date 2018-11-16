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