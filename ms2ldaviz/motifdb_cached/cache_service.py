"""
Cache service for motifdb to avoid database queries and reduce memory usage.
This module loads pre-computed motif data from JSON cache files.
"""

import json
import math
from pathlib import Path
from django.conf import settings


class MotifDBCacheService:
    """Service for managing cached motifdb data"""
    
    def __init__(self):
        self.cache_data = None
        self.cache_loaded = False
        self.load_cache()
    
    def load_cache(self):
        """Load the cache from JSON file"""
        # Determine cache file path
        cache_path = getattr(settings, 'MOTIFDB_CACHE_PATH', None)
        if not cache_path:
            # Default path
            cache_path = Path(__file__).parent / "cached_data" / "motifdb_cache.json"
        else:
            cache_path = Path(cache_path)
        
        if not cache_path.exists():
            print(f"Warning: Motifdb cache file not found at {cache_path}")
            print("Run 'python scripts/build_motifdb_cache.py' to build the cache")
            self.cache_data = {
                "motifsets": {},
                "motifset_data": {},
                "filtered_cache": {}
            }
            return
        
        try:
            with open(cache_path, 'r') as f:
                self.cache_data = json.load(f)
            self.cache_loaded = True
            print(f"Loaded motifdb cache from {cache_path}")
            print(f"  - {len(self.cache_data.get('motifsets', {}))} motifsets")
            print(f"  - {len(self.cache_data.get('filtered_cache', {}))} filtered combinations")
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache_data = {
                "motifsets": {},
                "motifset_data": {},
                "filtered_cache": {}
            }
    
    def get_all_motifsets(self):
        """Get list of all motifsets (compatible with list_motifsets view)"""
        result = {}
        for id_str, data in self.cache_data.get("motifsets", {}).items():
            result[data["name"]] = int(id_str)
        return result
    
    def get_motifset_by_id(self, motifset_id):
        """Get a single motifset's data (compatible with get_motifset view)"""
        id_str = str(motifset_id)
        if id_str not in self.cache_data.get("motifset_data", {}):
            return {}
        
        data = self.cache_data["motifset_data"][id_str]
        return data.get("motifs", {})
    
    def get_motifset_metadata(self, motifset_id):
        """Get metadata for a motifset (compatible with get_motifset_metadata view)"""
        id_str = str(motifset_id)
        if id_str not in self.cache_data.get("motifset_data", {}):
            return {}
        
        data = self.cache_data["motifset_data"][id_str]
        return data.get("metadata", {})
    
    def get_filtered_motifsets(self, motifset_id_list, filter_threshold=0.95):
        """
        Get filtered motifsets data (compatible with get_motifset_post view).
        First tries to find pre-computed filtered data, otherwise combines and filters on the fly.
        """
        # Convert to sorted string key for cache lookup
        id_list_sorted = sorted([int(x) for x in motifset_id_list])
        cache_key = f"{','.join(map(str, id_list_sorted))}_{filter_threshold}"
        
        # Check if we have pre-computed filtered data
        if cache_key in self.cache_data.get("filtered_cache", {}):
            cached = self.cache_data["filtered_cache"][cache_key]
            return {
                "motifs": cached["motifs"],
                "metadata": cached["metadata"]
            }
        
        # Otherwise, combine and filter on the fly
        combined_motifs = {}
        combined_metadata = {}
        
        for motifset_id in motifset_id_list:
            id_str = str(motifset_id)
            if id_str in self.cache_data.get("motifset_data", {}):
                data = self.cache_data["motifset_data"][id_str]
                combined_motifs.update(data.get("motifs", {}))
                combined_metadata.update(data.get("metadata", {}))
        
        # Apply filtering if requested
        if filter_threshold < 1.0 and combined_motifs:
            filter_obj = MotifFilter(combined_motifs, combined_metadata, filter_threshold)
            filtered_motifs, filtered_metadata = filter_obj.filter()
            return {
                "motifs": filtered_motifs,
                "metadata": filtered_metadata
            }
        
        return {
            "motifs": combined_motifs,
            "metadata": combined_metadata
        }
    
    def get_motif_by_id(self, motif_id):
        """Get a single motif's data (compatible with get_motif view)"""
        # Search through all motifsets for this motif
        for motifset_data in self.cache_data.get("motifset_data", {}).values():
            metadata = motifset_data.get("metadata", {})
            for motif_name, motif_meta in metadata.items():
                if motif_meta.get("motifdb_id") == motif_id:
                    # Found the motif, now get its spectrum
                    motifs = motifset_data.get("motifs", {})
                    if motif_name in motifs:
                        # Convert to format expected by get_motif view
                        output_list = []
                        for feature_name, probability in motifs[motif_name].items():
                            if feature_name.startswith('fragment'):
                                mz = float(feature_name.split('_')[1])
                                output_list.append((mz, probability))
                        return output_list
        return []


class MotifFilter:
    """Pure Python implementation of motif filtering (copied from views.py)"""
    
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
                    pos = spec_names.index(spec)
                    del spec_names[pos]
                self.input_metadata[current_spec]["merged"] = ",".join(spec_list)

        output_spectra = {}
        output_metadata = {}
        for spec in final_spec_list:
            output_spectra[spec] = self.input_spectra[spec]
            output_metadata[spec] = self.input_metadata[spec]
        
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


# Global instance
_cache_service = None


def get_cache_service():
    """Get or create the global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = MotifDBCacheService()
    return _cache_service