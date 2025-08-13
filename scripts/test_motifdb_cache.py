#!/usr/bin/env python
"""
Test script to verify the motifdb cache implementation matches the live server.
Run this with: python scripts/test_motifdb_cache.py
Make sure you're in the ms2ldaviz conda environment first.
"""

import json
import requests
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

MOTIFDB_SERVER_URL = "https://ms2lda.org/motifdb/"


def fetch_from_server(endpoint):
    """Fetch data directly from the production server"""
    url = MOTIFDB_SERVER_URL + endpoint
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def fetch_filtered_from_server(motifset_ids, filter_threshold=0.95):
    """Fetch filtered data from server using POST"""
    url = MOTIFDB_SERVER_URL + "get_motifset/"
    
    # Get CSRF token first
    init_url = MOTIFDB_SERVER_URL + "initialise_api/"
    session = requests.Session()
    token_response = session.get(init_url)
    token = token_response.json()["token"]
    
    # Prepare request data
    data = {
        "csrfmiddlewaretoken": token,
        "motifset_id_list": motifset_ids,
        "filter": "True",
        "filter_threshold": filter_threshold
    }
    
    response = session.post(url, data=data)
    response.raise_for_status()
    return response.json()


def compare_results(cached_data, server_data, description):
    """Compare cached and server results"""
    if isinstance(cached_data, dict) and isinstance(server_data, dict):
        cached_keys = set(cached_data.keys())
        server_keys = set(server_data.keys())
        
        if cached_keys == server_keys:
            print(f"✓ {description}: Keys match ({len(cached_keys)} items)")
            return True
        else:
            print(f"✗ {description}: Key mismatch")
            print(f"  Cached has {len(cached_keys)} keys")
            print(f"  Server has {len(server_keys)} keys")
            missing_in_cache = server_keys - cached_keys
            extra_in_cache = cached_keys - server_keys
            if missing_in_cache:
                print(f"  Missing in cache: {list(missing_in_cache)[:5]}")
            if extra_in_cache:
                print(f"  Extra in cache: {list(extra_in_cache)[:5]}")
            return False
    else:
        # For simple comparisons
        if cached_data == server_data:
            print(f"✓ {description}: Data matches")
            return True
        else:
            print(f"✗ {description}: Data mismatch")
            return False


def main():
    print("="*60)
    print("MOTIFDB CACHE VALIDATION TEST")
    print("="*60)
    print("This test compares cached data with live server data\n")
    
    # Check if cache file exists
    cache_file = Path(parent_dir) / "ms2ldaviz" / "motifdb" / "cached_data" / "motifdb_cache.json"
    if not cache_file.exists():
        print(f"✗ Cache file not found at: {cache_file}")
        print("  Run: python scripts/build_motifdb_cache.py")
        return
    
    print(f"Loading cache from: {cache_file}")
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)
    print(f"Cache file size: {file_size:.2f} MB\n")
    
    # Test 1: Compare motifset list
    print("Test 1: Comparing motifset list")
    print("-" * 40)
    try:
        server_motifsets = fetch_from_server("list_motifsets/")
        
        # Convert cache format to match server format
        cached_motifsets = {}
        for id_str, data in cache_data.get("motifsets", {}).items():
            cached_motifsets[data["name"]] = int(id_str)
        
        compare_results(cached_motifsets, server_motifsets, "Motifset list")
    except Exception as e:
        print(f"✗ Error comparing motifset list: {e}")
    
    # Test 2: Compare individual motifset data (test first 3)
    print("\nTest 2: Comparing individual motifset data")
    print("-" * 40)
    test_ids = list(cache_data.get("motifsets", {}).keys())[:3]
    
    for motifset_id in test_ids:
        try:
            motifset_name = cache_data["motifsets"][motifset_id]["name"]
            print(f"\nChecking motifset '{motifset_name}' (ID: {motifset_id})...")
            
            # Get data from server
            server_single = fetch_from_server(f"get_motifset/{motifset_id}/")
            
            # Get data from cache
            cached_single = cache_data.get("motifset_data", {}).get(motifset_id, {}).get("motifs", {})
            
            # Compare
            if len(cached_single) == len(server_single):
                print(f"  ✓ Motif count matches: {len(cached_single)} motifs")
                
                # Check a few motifs in detail
                sample_motifs = list(cached_single.keys())[:2]
                all_match = True
                for motif_name in sample_motifs:
                    if motif_name in server_single:
                        cached_features = cached_single[motif_name]
                        server_features = server_single[motif_name]
                        if cached_features == server_features:
                            print(f"  ✓ Motif '{motif_name}' data matches")
                        else:
                            print(f"  ✗ Motif '{motif_name}' data differs")
                            all_match = False
                    else:
                        print(f"  ✗ Motif '{motif_name}' missing from server")
                        all_match = False
            else:
                print(f"  ✗ Motif count mismatch: cached={len(cached_single)}, server={len(server_single)}")
        
        except Exception as e:
            print(f"  ✗ Error checking motifset {motifset_id}: {e}")
    
    # Test 3: Compare filtered results
    print("\nTest 3: Comparing filtered results")
    print("-" * 40)
    test_combinations = [
        ([2], 0.95),  # GNPS only
        ([1, 2], 0.95),  # Urine + GNPS
    ]
    
    for combo, threshold in test_combinations:
        combo_str = f"[{','.join(map(str, combo))}]"
        print(f"\nChecking filtered combination {combo_str} at threshold {threshold}...")
        
        try:
            # Get from server
            server_filtered = fetch_filtered_from_server(combo, threshold)
            
            # Get from cache
            cache_key = f"{','.join(map(str, sorted(combo)))}_{threshold}"
            if cache_key in cache_data.get("filtered_cache", {}):
                cached_filtered = cache_data["filtered_cache"][cache_key]
                
                # Compare motif counts
                server_motif_count = len(server_filtered.get("motifs", {}))
                cached_motif_count = len(cached_filtered.get("motifs", {}))
                
                if server_motif_count == cached_motif_count:
                    print(f"  ✓ Filtered motif count matches: {cached_motif_count} motifs")
                else:
                    print(f"  ✗ Filtered motif count mismatch: cached={cached_motif_count}, server={server_motif_count}")
                
                # Check if the motif names match
                server_motif_names = set(server_filtered.get("motifs", {}).keys())
                cached_motif_names = set(cached_filtered.get("motifs", {}).keys())
                
                if server_motif_names == cached_motif_names:
                    print(f"  ✓ Filtered motif names match")
                else:
                    missing = server_motif_names - cached_motif_names
                    extra = cached_motif_names - server_motif_names
                    if missing:
                        print(f"  ✗ Missing motifs in cache: {list(missing)[:3]}")
                    if extra:
                        print(f"  ✗ Extra motifs in cache: {list(extra)[:3]}")
            else:
                print(f"  ⚠ No pre-computed cache for {cache_key}")
                print(f"    (Will be computed on-the-fly when requested)")
        
        except Exception as e:
            print(f"  ✗ Error checking filtered combination: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("CACHE VALIDATION SUMMARY")
    print("="*60)
    print(f"Cache contains:")
    print(f"  - {len(cache_data.get('motifsets', {}))} motifsets")
    print(f"  - {len(cache_data.get('motifset_data', {}))} cached datasets")
    print(f"  - {len(cache_data.get('filtered_cache', {}))} pre-computed filtered combinations")
    print("\nNote: Some differences may be expected if the server data")
    print("has been updated since the cache was built.")
    print("\nTo rebuild cache: python scripts/build_motifdb_cache.py")


if __name__ == "__main__":
    main()