import requests

server_url = 'https://ms2lda.org/motifdb/'
# server_url = 'http://localhost/motifdb/'

"""Grabbing the latest Motifs from MS2LDA"""
motifset_dict = requests.get(server_url + 'list_motifsets/').json()
db_list = []
db_list.append(2)
db_list.append(4)
db_list.append(1)
db_list.append(3)
db_list.append(5)
db_list.append(6)
db_list.append(16)

# Acquire motifset from MS2LDA.org
client = requests.session()

# no longer needed
# token_output = client.get(server_url + 'initialise_api/', verify=False).json()
# token = token_output['token']
# data = {'csrfmiddlewaretoken': token}

data = {}
data['motifset_id_list'] = db_list
data['filter'] = 'True'

response = client.post(server_url + 'get_motifset/', data=data)
json_output = response.json()
# print(json_output)

assert response.status_code == 200
assert len(json_output['motifs']) > 0
assert len(json_output['metadata']) > 0
print('Success!')
