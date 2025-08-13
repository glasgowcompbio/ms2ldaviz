#!/usr/bin/env python
"""
Build a cached version of all motifdb data to eliminate database queries.
This script fetches all motif sets from the production server and pre-computes
filtered versions at common thresholds.
"""

import json
import math
import os
import requests
import sys
from pathlib import Path

# Add parent directory to path to import ms2ldaviz modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

MOTIFDB_SERVER_URL = "https://ms2lda.org/motifdb/"


class MotifFilter:
    """Pure Python implementation of motif filtering without numpy"""
    
    def __init__(self, spectra, metadata, threshold=0.95):
        self.input_spectra = spectra
        self.input_metadata = metadata
        self.threshold = threshold

    def filter(self):
        """Greedy filtering - merge similar motifs"""
        spec_names = sorted(self.input_metadata.keys())
        final_spec_list = []
        
        while len(spec_names) > 0:
            current_spec = spec_names[0]
            final_spec_list.append(current_spec)
            del spec_names[0]
            merge_list = []
            
            for spec in spec_names:
                sim = self.compute_similarity(current_spec, spec)
                if sim >= self.threshold:
                    merge_list.append((spec, sim))
            
            if len(merge_list) > 0:
                spec_list = []
                for spec, sim in merge_list:
                    spec_list.append(spec)
                    print(f"  Merging: {current_spec} and {spec} ({sim:.3f})")
                    pos = spec_names.index(spec)
                    del spec_names[pos]
                self.input_metadata[current_spec]["merged"] = ",".join(spec_list)

        output_spectra = {}
        output_metadata = {}
        for spec in final_spec_list:
            output_spectra[spec] = self.input_spectra[spec]
            output_metadata[spec] = self.input_metadata[spec]
        
        print(f"  After merging, {len(output_spectra)} motifs remain")
        return output_spectra, output_metadata

    def compute_similarity(self, k, k2):
        """Compute cosine similarity using pure Python"""
        prod = 0
        i1 = 0
        
        for mz, intensity in self.input_spectra[k].items():
            i1 += intensity ** 2
            for mz2, intensity2 in self.input_spectra[k2].items():
                if mz == mz2:
                    prod += intensity * intensity2
        
        i2 = sum([i**2 for i in self.input_spectra[k2].values()])
        return prod / (math.sqrt(i1) * math.sqrt(i2))


def get_motifset_list():
    """Fetch list of all available motif sets"""
    url = MOTIFDB_SERVER_URL + "list_motifsets/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def fetch_motifset_data(motifset_ids, filter_threshold=None):
    """Fetch motif data for given motifset IDs with optional filtering"""
    url = MOTIFDB_SERVER_URL + "get_motifset/"
    
    # Get CSRF token first
    init_url = MOTIFDB_SERVER_URL + "initialise_api/"
    session = requests.Session()
    token_response = session.get(init_url)
    token = token_response.json()["token"]
    
    # Prepare request data
    data = {
        "csrfmiddlewaretoken": token,
        "motifset_id_list": motifset_ids
    }
    
    if filter_threshold is not None:
        data["filter"] = "True"
        data["filter_threshold"] = filter_threshold
    else:
        data["filter"] = "False"
    
    # Fetch data
    response = session.post(url, data=data)
    response.raise_for_status()
    return response.json()


def build_cache():
    """Build the complete cache of motifdb data"""
    print("Building motifdb cache...")
    print(f"Fetching from: {MOTIFDB_SERVER_URL}")
    
    # Get list of all motif sets
    print("\n1. Fetching motif set list...")
    motifset_list = get_motifset_list()
    print(f"   Found {len(motifset_list)} motif sets")
    
    cache = {
        "motifsets": {},
        "motifset_data": {},
        "filtered_cache": {}
    }
    
    # Store motifset info
    for name, id_val in motifset_list.items():
        cache["motifsets"][str(id_val)] = {
            "id": id_val,
            "name": name
        }
    
    # Fetch individual motifset data (unfiltered)
    print("\n2. Fetching individual motifset data...")
    for name, id_val in motifset_list.items():
        print(f"   Fetching {name} (ID: {id_val})...")
        try:
            data = fetch_motifset_data([id_val], filter_threshold=None)
            cache["motifset_data"][str(id_val)] = {
                "motifs": data.get("motifs", {}),
                "metadata": data.get("metadata", {})
            }
            print(f"     Got {len(data.get('motifs', {}))} motifs")
        except Exception as e:
            print(f"     ERROR: {e}")
            cache["motifset_data"][str(id_val)] = {
                "motifs": {},
                "metadata": {}
            }
    
    # Pre-compute filtered versions for common combinations
    print("\n3. Pre-computing filtered combinations...")
    
    # Common motifset combinations used by GNPS
    common_combinations = [
        [1, 2],     # Urine + GNPS
        [2],        # GNPS only
        [4],        # Massbank only
        [1],        # Urine only
        [2, 4],     # GNPS + Massbank
        [1, 2, 4],  # Urine + GNPS + Massbank
        [3],        # Euphorbia
        [5],        # Rhamnaceae
        [6],        # Strep/Salin
        [16],       # Photorhabdus
    ]
    
    # Common filter thresholds
    thresholds = [0.95, 0.90, 0.85]
    
    for combo in common_combinations:
        # Skip if any ID doesn't exist
        if not all(str(id_val) in cache["motifsets"] for id_val in combo):
            continue
            
        combo_name = ",".join(map(str, sorted(combo)))
        
        for threshold in thresholds:
            cache_key = f"{combo_name}_{threshold}"
            print(f"   Computing filtered cache for motifsets [{combo_name}] at threshold {threshold}...")
            
            try:
                # Combine motifs from multiple sets
                combined_motifs = {}
                combined_metadata = {}
                
                for id_val in combo:
                    id_str = str(id_val)
                    if id_str in cache["motifset_data"]:
                        combined_motifs.update(cache["motifset_data"][id_str]["motifs"])
                        combined_metadata.update(cache["motifset_data"][id_str]["metadata"])
                
                # Apply filtering
                if combined_motifs:
                    filter_obj = MotifFilter(combined_motifs, combined_metadata, threshold)
                    filtered_motifs, filtered_metadata = filter_obj.filter()
                    
                    cache["filtered_cache"][cache_key] = {
                        "motifs": filtered_motifs,
                        "metadata": filtered_metadata
                    }
                    print(f"     Cached {len(filtered_motifs)} motifs after filtering")
                
            except Exception as e:
                print(f"     ERROR computing filtered cache: {e}")
    
    return cache


def main():
    """Main function to build and save the cache"""
    # Build the cache
    cache = build_cache()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "ms2ldaviz" / "motifdb" / "cached_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cache to JSON
    output_file = output_dir / "motifdb_cache.json"
    print(f"\n4. Saving cache to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CACHE BUILD COMPLETE")
    print("="*60)
    print(f"Total motif sets: {len(cache['motifsets'])}")
    print(f"Total cached data sets: {len(cache['motifset_data'])}")
    print(f"Total filtered combinations: {len(cache['filtered_cache'])}")
    
    # Calculate file size
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"Cache file size: {file_size:.2f} MB")
    
    print("\nCache is now ready to use!")
    print("The motifdb API will automatically use this cached data.")


if __name__ == "__main__":
    main()