#!/usr/bin/env python3
"""
Verify that cached motifdb endpoints return identical results to the remote server.
"""

import json
import requests
import sys
from typing import Dict, Any, List, Tuple


def normalize_json(data: Any) -> Any:
    """Normalize JSON data for comparison by sorting keys and lists."""
    if isinstance(data, dict):
        return {k: normalize_json(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        # Sort lists of dicts by their sorted representation
        if data and isinstance(data[0], dict):
            return sorted([normalize_json(item) for item in data], 
                         key=lambda x: json.dumps(x, sort_keys=True))
        else:
            return sorted([normalize_json(item) for item in data])
    else:
        return data


def compare_endpoints(local_url: str, remote_url: str, endpoint: str) -> Tuple[bool, str]:
    """Compare a single endpoint between local and remote servers."""
    try:
        # Fetch from local server
        local_response = requests.get(f"{local_url}{endpoint}", timeout=10)
        local_response.raise_for_status()
        local_data = normalize_json(local_response.json())
        
        # Fetch from remote server
        remote_response = requests.get(f"{remote_url}{endpoint}", timeout=10)
        remote_response.raise_for_status()
        remote_data = normalize_json(remote_response.json())
        
        # Compare
        if local_data == remote_data:
            return True, "MATCH"
        else:
            # Try to provide more detail about the difference
            local_json = json.dumps(local_data, sort_keys=True, indent=2)
            remote_json = json.dumps(remote_data, sort_keys=True, indent=2)
            
            if len(local_json) != len(remote_json):
                return False, f"Size mismatch: local={len(local_json)}, remote={len(remote_json)}"
            else:
                # Find first difference
                for i, (l, r) in enumerate(zip(local_json, remote_json)):
                    if l != r:
                        context_start = max(0, i-50)
                        context_end = min(len(local_json), i+50)
                        return False, f"Difference at position {i}: local=...{local_json[context_start:context_end]}..., remote=...{remote_json[context_start:context_end]}..."
                return False, "Content differs"
                
    except requests.RequestException as e:
        return False, f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return False, f"JSON decode failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def test_post_endpoint(local_url: str, remote_url: str) -> Tuple[bool, str]:
    """Test the POST endpoint for filtered motifsets."""
    endpoint = "/motifdb/get_motifset/"
    
    # Test parameters from the original GNPS workflow
    test_cases = [
        {
            'motifset_id_list': ['3', '5'],
            'filter': 'True',
            'filter_threshold': '0.95'
        },
        {
            'motifset_id_list': ['3', '5'],
            'filter': 'True',
            'filter_threshold': '0.90'
        },
        {
            'motifset_id_list': ['3', '5'],
            'filter': 'False'
        }
    ]
    
    all_match = True
    messages = []
    
    for i, params in enumerate(test_cases):
        try:
            # Test local
            local_response = requests.post(f"{local_url}{endpoint}", data=params, timeout=10)
            local_response.raise_for_status()
            local_data = normalize_json(local_response.json())
            
            # Test remote
            remote_response = requests.post(f"{remote_url}{endpoint}", data=params, timeout=10)
            remote_response.raise_for_status()
            remote_data = normalize_json(remote_response.json())
            
            if local_data == remote_data:
                messages.append(f"  Test case {i+1}: MATCH")
            else:
                all_match = False
                messages.append(f"  Test case {i+1}: MISMATCH")
                
                # Count motifs for debugging
                local_count = len(local_data.get('motifs', {}))
                remote_count = len(remote_data.get('motifs', {}))
                messages.append(f"    Motif count: local={local_count}, remote={remote_count}")
                
        except Exception as e:
            all_match = False
            messages.append(f"  Test case {i+1}: ERROR - {e}")
    
    return all_match, "\n".join(messages)


def main():
    # Server URLs
    LOCAL_URL = "http://localhost:8000"
    REMOTE_URL = "https://ms2lda.org"
    
    print("=" * 60)
    print("Verifying Cached MotifDB Endpoints")
    print("=" * 60)
    print(f"Local server:  {LOCAL_URL}")
    print(f"Remote server: {REMOTE_URL}")
    print()
    
    # Check if local server is running
    try:
        requests.get(f"{LOCAL_URL}/", timeout=2)
    except requests.RequestException:
        print("ERROR: Local server is not running!")
        print("Please start the server with:")
        print("  cd ms2ldaviz && pipenv run python manage.py runserver")
        sys.exit(1)
    
    # Test GET endpoints
    get_endpoints = [
        "/motifdb/list_motifsets/",
        "/motifdb/get_motifset/3/",
        "/motifdb/get_motifset/5/",
        "/motifdb/get_motif/869/",
        "/motifdb/get_motif/870/",
        "/motifdb/get_motifset_metadata/3/",
        "/motifdb/get_motifset_metadata/5/",
    ]
    
    print("Testing GET endpoints:")
    print("-" * 40)
    
    all_pass = True
    for endpoint in get_endpoints:
        match, message = compare_endpoints(LOCAL_URL, REMOTE_URL, endpoint)
        status = "✓" if match else "✗"
        print(f"{status} {endpoint}: {message}")
        if not match:
            all_pass = False
    
    print()
    print("Testing POST endpoint (filtered motifsets):")
    print("-" * 40)
    
    post_match, post_message = test_post_endpoint(LOCAL_URL, REMOTE_URL)
    status = "✓" if post_match else "✗"
    print(f"{status} /motifdb/get_motifset/ (POST):")
    print(post_message)
    if not post_match:
        all_pass = False
    
    print()
    print("=" * 60)
    if all_pass:
        print("✓ SUCCESS: All cached endpoints match the remote server!")
    else:
        print("✗ FAILURE: Some endpoints do not match.")
        print("\nThis could be due to:")
        print("1. Cache data being out of date")
        print("2. Remote server having been updated")
        print("3. Differences in data normalization")
        print("\nTo rebuild the cache, run:")
        print("  python scripts/build_motifdb_cache.py")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())