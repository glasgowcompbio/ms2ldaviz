import requests

server_url = "http://ms2lda.org"
# server_url = "http://127.0.0.1:8000"

# Get the list of motif sets
output = requests.get(server_url + '/motifdb/list_motifsets')
motifset_list = output.json()

# Get a token
url = server_url + '/motifdb/initialise_api'
client = requests.session()
token = client.get(url).json()['token']

url = server_url + '/motifdb/get_motifset/'
data = {'csrfmiddlewaretoken': token}

# specify the name of motifsets of interest in motifset_list
massbank_motifset = 'Massbank library derived Mass2Motifs'
gnps_motifset = 'GNPS library derived Mass2Motifs'

massbank_id = motifset_list[massbank_motifset]
gnps_id = motifset_list[gnps_motifset]

# example - get the massbank and gnps motifsets
data['motifset_id_list'] = [massbank_id, gnps_id]
data['filter'] = "True"
data['filter_threshold'] = 0.95 # Default value - not required
output = client.post(url, data=data).json()
print('Retrieved', len(output['motifs']), 'motifs', len(output['metadata']))

# example - get the massbank metadata only
url = server_url + '/motifdb/get_motifset_metadata/{}/'.format(massbank_id)
output = requests.get(url)
motif_metadata = output.json()
for k in motif_metadata.keys()[:5]:
    print(k)
    print(motif_metadata[k])