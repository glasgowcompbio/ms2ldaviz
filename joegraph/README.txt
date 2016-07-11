graph.html is the D3 visualisation to show the network structure defined in graph.json

Chrome does not allow a local web page to load another file in the filesystem,
so to run this example, we need to start a web server. Not sure about safari ..

$ python -m SimpleHTTPServer 9090

graph.json is exported from networkx, see a minimal example below ..

G = nx.Graph()

# add document node
doc_id = 570
node_name = 'doc_254.11334_520.976'
node_group = 1
node_score = 0 # unused ?
node_size = 10
node_type = 'square'
node_peakid = '17091'

G.add_node(node_id, name=node_name, group=node_group, in_degree=0, size=node_size, score=node_score,
           type=node_type, special=False, peakid=node_peakid)

# add topic node
topic_id = 791
node_name = 'motif_2'
node_group = 2
node_size = 1145
node_score = 1 # unused ?
node_type = "circle"
in_degree = G.degree(node_id)

G.add_node(node_id, name=node_name, group=node_group, in_degree=in_degree, size=node_size, score=node_score,
           type=node_type, special=False, peakid=node_peakid)

weight = 1.0
G.add_edge(doc_id, topic_id, weight=weight)

from networkx.readwrite import json_graph
json_out = json_graph.node_link_data(G) # node-link format to serialize