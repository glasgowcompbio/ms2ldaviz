#%%
import requests
server_url = "http://127.0.0.1:8000"

#%%
output = requests.get(server_url + '/motifdb/list_motifsets')
motifset_list = output.json()
#%%
# example - get the massbank ones
# url = server_url+'/motifdb/get_motifset/{}/'.format(motifset_list['massbank_binned_005'])
# output = requests.get(url)
# motif_set = output.json()
# for k in motif_set.keys()[:2]:
#     print k
#     print motif_set[k]
#%%
url = server_url + '/motifdb/get_motifset'
client = requests.session()
client.get(url)
csrftoken = client.cookies['csrftoken']

data = {'motifset_id':motifset_list['massbank_binned_005']}
output = client.post(url,data = data, headers=dict(Referer=url))
print output.text
#%%
# example - get the massbank metadata
url = server_url + '/motifdb/get_motifset_metadata/{}/'.format(motifset_list['massbank_binned_005'])
output = requests.get(url)
motif_metadata = output.json()
for k in motif_metadata.keys()[:5]:
    print k
    print motif_metadata[k]
#%%

#%%
