# -*- coding: utf-8 -*-
"""
Cached MotifDB views - serves JSON data from cache without database dependencies
"""

from django.shortcuts import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import cache_page
from django.conf import settings

import json

from motifdb_cached.cache_service import get_cache_service


def index(request):
    """MotifDB web interface is disabled when using cached data"""
    return HttpResponse("MotifDB web interface is disabled (using cached data)", status=503)


def motif_set(request, motif_set_id):
    """MotifDB web interface is disabled when using cached data"""
    return HttpResponse("MotifDB web interface is disabled (using cached data)", status=503)


def motif(request, motif_id):
    """MotifDB web interface is disabled when using cached data"""
    return HttpResponse("MotifDB web interface is disabled (using cached data)", status=503)


def get_motif(request, motif_id):
    """Get a single motif's data from cache"""
    cache_service = get_cache_service()
    output_list = cache_service.get_motif_by_id(int(motif_id))
    return HttpResponse(json.dumps(output_list), content_type='application/json')


def update_annotation(request, motif_id):
    """Annotation updates are disabled when using cached data"""
    return HttpResponse(json.dumps({"error": "Updates are disabled in cached mode"}), 
                       content_type='application/json', status=503)


def start_motif_matching(request, experiment_id):
    """Motif matching is disabled when using cached data"""
    return HttpResponse(json.dumps({"error": "Motif matching is disabled in cached mode"}), 
                       content_type='application/json', status=503)


def list_motifsets(request):
    """List all motifsets from cache"""
    cache_service = get_cache_service()
    output = cache_service.get_all_motifsets()
    return HttpResponse(json.dumps(output), content_type='application/json')


@cache_page(settings.DEFAULT_CACHE_TIMEOUT)
def get_motifset(request, motifset_id):
    """Get motifset data from cache"""
    cache_service = get_cache_service()
    output_motifs = cache_service.get_motifset_by_id(motifset_id)
    return HttpResponse(json.dumps(output_motifs), content_type='application/json')


@csrf_exempt
def get_motifset_post(request):
    """Get filtered motifset data from cache"""
    motifset_id_list = request.POST.getlist('motifset_id_list')
    do_filter = request.POST.get('filter', "False") == "True"
    filter_threshold = float(request.POST.get('filter_threshold', 0.95))
    
    cache_service = get_cache_service()
    
    if do_filter:
        result = cache_service.get_filtered_motifsets(motifset_id_list, filter_threshold)
    else:
        result = cache_service.get_filtered_motifsets(motifset_id_list, 1.0)
    
    return HttpResponse(json.dumps(result["motifs"]), content_type='application/json')


@cache_page(settings.DEFAULT_CACHE_TIMEOUT)
def get_motifset_metadata(request, motifset_id):
    """Get motifset metadata from cache"""
    cache_service = get_cache_service()
    output_motifs = cache_service.get_motifset_metadata(motifset_id)
    return HttpResponse(json.dumps(output_motifs), content_type='application/json')


def initialise_api(request):
    """API initialization endpoint"""
    return HttpResponse(json.dumps({"status": "ok", "mode": "cached"}), 
                       content_type='application/json')


def create_motifset(request):
    """Creating motifsets is disabled when using cached data"""
    return HttpResponse(json.dumps({"error": "Creating motifsets is disabled in cached mode"}), 
                       content_type='application/json', status=503)


def choose_motifs(request, motif_set_id, experiment_id):
    """Choosing motifs is disabled when using cached data"""
    return HttpResponse("Choosing motifs is disabled (using cached data)", status=503)


def edit_motifset_metadata(request, motif_set_id):
    """Editing metadata is disabled when using cached data"""
    return HttpResponse("Editing metadata is disabled (using cached data)", status=503)