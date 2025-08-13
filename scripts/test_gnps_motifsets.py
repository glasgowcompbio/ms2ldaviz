#!/usr/bin/env python3
"""
Test all motifsets used by GNPS workflow against the remote server.
"""

import json
import requests
import sys

# Motifsets used by GNPS (from pySubstructures constants)
GNPS_MOTIFSETS = {
    1: "Urine derived Mass2Motifs",
    2: "GNPS library derived Mass2Motifs",
    3: "Euphorbia Plant Mass2Motifs",
    4: "Massbank library derived Mass2Motifs",
    5: "Rhamnaceae Plant Mass2Motifs",
    6: "Streptomyces and Salinispora Mass2Motifs",
    16: "Photorhabdus and Xenorhabdus Mass2Motifs"
}

def normalize_json(data):
    """Normalize JSON for comparison."""
    if isinstance(data, dict):
        return {k: normalize_json(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            return sorted([normalize_json(item) for item in data], 
                         key=lambda x: json.dumps(x, sort_keys=True))
        else:
            return sorted([normalize_json(item) for item in data])
    else:
        return data

def test_get_motifset(motifset_id, name):
    """Test GET endpoint for a single motifset."""
    endpoint = f"/motifdb/get_motifset/{motifset_id}/"
    
    try:
        local_resp = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
        remote_resp = requests.get(f"https://ms2lda.org{endpoint}", timeout=10)
        
        if local_resp.status_code != 200:
            return False, f"Local server error: {local_resp.status_code}"
        if remote_resp.status_code != 200:
            return False, f"Remote server error: {remote_resp.status_code}"
            
        local_data = normalize_json(local_resp.json())
        remote_data = normalize_json(remote_resp.json())
        
        if local_data == remote_data:
            # Count motifs for info
            motif_count = len(local_data.get('motifs', {}))
            return True, f"MATCH ({motif_count} motifs)"
        else:
            local_count = len(local_data.get('motifs', {}))
            remote_count = len(remote_data.get('motifs', {}))
            return False, f"MISMATCH (local={local_count}, remote={remote_count} motifs)"
            
    except Exception as e:
        return False, f"ERROR: {e}"

def test_post_filtered(motifset_ids, filter_threshold):
    """Test POST endpoint with filtering."""
    endpoint = "/motifdb/get_motifset/"
    params = {
        'motifset_id_list': [str(id) for id in motifset_ids],
        'filter': 'True',
        'filter_threshold': str(filter_threshold)
    }
    
    try:
        local_resp = requests.post(f"http://localhost:8000{endpoint}", data=params, timeout=30)
        remote_resp = requests.post(f"https://ms2lda.org{endpoint}", data=params, timeout=30)
        
        if local_resp.status_code != 200:
            return False, f"Local server error: {local_resp.status_code}"
        if remote_resp.status_code != 200:
            return False, f"Remote server error: {remote_resp.status_code}"
            
        local_data = normalize_json(local_resp.json())
        remote_data = normalize_json(remote_resp.json())
        
        if local_data == remote_data:
            motif_count = len(local_data.get('motifs', {}))
            return True, f"MATCH ({motif_count} motifs after filtering)"
        else:
            local_count = len(local_data.get('motifs', {}))
            remote_count = len(remote_data.get('motifs', {}))
            return False, f"MISMATCH (local={local_count}, remote={remote_count} motifs)"
            
    except Exception as e:
        return False, f"ERROR: {e}"

def main():
    print("=" * 70)
    print("Testing GNPS MotifDB Endpoints")
    print("=" * 70)
    
    # Check if local server is running
    try:
        requests.get("http://localhost:8000/", timeout=2)
    except:
        print("ERROR: Local server not running!")
        print("Start it with: cd ms2ldaviz && pipenv run python manage.py runserver")
        return 1
    
    all_pass = True
    
    # Test individual motifsets
    print("\n1. Testing individual motifset GET endpoints:")
    print("-" * 50)
    for motifset_id, name in GNPS_MOTIFSETS.items():
        success, message = test_get_motifset(motifset_id, name)
        status = "✓" if success else "✗"
        print(f"{status} ID {motifset_id:2d}: {name[:40]:40s} - {message}")
        if not success:
            all_pass = False
    
    # Test combined POST requests (typical GNPS usage)
    print("\n2. Testing combined POST requests (with filtering):")
    print("-" * 50)
    
    test_combinations = [
        # Test plant motifsets together (common use case)
        ([3, 5], 0.95, "Euphorbia + Rhamnaceae @ 0.95"),
        ([3, 5], 0.90, "Euphorbia + Rhamnaceae @ 0.90"),
        
        # Test library motifsets
        ([2, 4], 0.95, "GNPS + Massbank libraries @ 0.95"),
        
        # Test all 7 motifsets together
        (list(GNPS_MOTIFSETS.keys()), 0.95, "All 7 GNPS motifsets @ 0.95"),
        (list(GNPS_MOTIFSETS.keys()), 0.90, "All 7 GNPS motifsets @ 0.90"),
    ]
    
    for motifset_ids, threshold, description in test_combinations:
        success, message = test_post_filtered(motifset_ids, threshold)
        status = "✓" if success else "✗"
        print(f"{status} {description:35s} - {message}")
        if not success:
            all_pass = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ SUCCESS: All GNPS motifset endpoints match the remote server!")
    else:
        print("✗ FAILURE: Some endpoints do not match.")
        print("\nNote: Minor differences in metadata formatting are expected")
        print("The important thing is that the motif data itself matches.")
    print("=" * 70)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())