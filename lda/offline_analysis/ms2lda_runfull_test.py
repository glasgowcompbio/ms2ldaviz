import json

# try:
#     redis_connection = redis.Redis(host='dorresteinappshub.ucsd.edu', port=6378, db=0)
# except:
#     redis_connection = None
redis_connection = None


def acquire_motifdb(db_list):
    db_list_key = json.dumps(db_list)
    if redis_connection is not None:
        if redis_connection.exists(db_list_key):
            cached_data = json.loads(redis_connection.get(db_list_key))
            return cached_data["motifdb_spectra"], cached_data["motifdb_metadata"], set(cached_data["motifdb_features"])

    client = requests.session()
    token_output = client.get(server_url + 'initialise_api/').json()
    token = token_output['token']
    data = {'csrfmiddlewaretoken': token}
    data['motifset_id_list'] = db_list
    data['filter'] = 'True'

    output = client.post(server_url + 'get_motifset/', data=data).json()
    motifdb_spectra = output['motifs']
    motifdb_metadata = output['metadata']
    motifdb_features = set()
    for m, spec in motifdb_spectra.items():
        for f in spec:
            motifdb_features.add(f)

    # Trying to cache
    if redis_connection is not None:
        data_cache = {}
        data_cache["motifdb_spectra"] = motifdb_spectra
        data_cache["motifdb_metadata"] = motifdb_metadata
        data_cache["motifdb_features"] = list(motifdb_features)

        redis_connection.set(db_list_key, json.dumps(data_cache))

    return motifdb_spectra, motifdb_metadata, motifdb_features


"""Grabbing the latest Motifs from MS2LDA"""
import requests

server_url = 'http://ms2lda.org/motifdb/'
server_url = 'http://localhost:8000/motifdb/'

motifset_dict = requests.get(server_url + 'list_motifsets/').json()
# db_list = ['gnps_binned_005']  # Can update this later with multiple motif sets
db_list = []
# db_list.append(2)
# db_list.append(4)
# db_list.append(1)
# db_list.append(3)
# db_list.append(5)
# db_list.append(6)
# db_list.append(16)
db_list = list(set(db_list))

# Acquire motifset from MS2LDA.org
motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifdb(db_list)
