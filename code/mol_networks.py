# coding=utf8
# python code to do some molecular networking

from pprint import PrettyPrinter
import numpy as np
def modified_cosine(spec1,spec2,tol = 0.3):
	# compute the modified cosine
	# spec1 and spec2 are lists of (mz,i) tuples
	# spectra *must* be sorted in ascending mz
	# and normalised such that biggest peak = 100
	pp = PrettyPrinter()

	pp.pprint(spec1)
	pp.pprint(spec2)

	norm1 = 0.0
	norm2 = 0.0

	spec2_start = 0 # start of range to search in
	total = 0.0
	# finished = False
	# for peak in spec1:
	# 	biggest = 0.0
	# 	while peak[0]-spec2[spec2_start][0] > tol:
	# 		spec2_start += 1
	# 		if spec2_start == len(spec2):
	# 			finished = True
	# 			break
	# 	if finished:
	# 		break

	# 	spec2_end = spec2_start

	# 	while spec2[spec2_end][0] - peak[0] < tol:
	# 		spec2_end += 1
	# 		if spec2_end == len(spec2):
	# 			break

	# 	for peak2 in spec2[spec2_start:spec2_end]:
	# 		print peak2
	# 	print peak,spec2[spec2_start],spec2[spec2_end]
	# 	break
	total = 0.0
	for peak1 in spec1:
		biggest = 0.0
		for peak2 in spec2:
			if np.abs(peak1[0]-peak2[0]) <= tol:
				biggest = max(biggest,np.sqrt(peak1[1])*np.sqrt(peak2[1]))
				# biggest = peak1[1]*peak2[1]
		total += biggest
	norm1 = np.sqrt(sum([x[1] for x in spec1]))
	norm2 = np.sqrt(sum([x[1] for x in spec2]))

	return 1.0*(total)/(norm1*norm2)



def create_edge_dict(scores,ms1,min_frag_overlap = 6,min_score = 0.7):
	edge_dict = {}
	for score_row in scores:
		mol1 = ms1[score_row[0]]
		mol2 = ms1[score_row[1]]
		score = score_row[2]
		overlap = score_row[3]
		if overlap >= min_frag_overlap and score >= min_score:
			if not mol1 in edge_dict:
				edge_dict[mol1] = []
			if not mol2 in edge_dict:
				edge_dict[mol2] = []
			edge_dict[mol1].append((mol2,score))
			edge_dict[mol2].append((mol1,score))

	for mol,edges in edge_dict.items():
		sorted_edges = sorted(edges,key = lambda x: x[1],reverse = True)
		edge_dict[mol] = sorted_edges
	return edge_dict


def top_k_filter(edge_dict,k = 10):
	for mol,edges in edge_dict.items():
		if len(edges) > k:
			edges = edges[:k]
			edge_dict[mol] = edges
	# ensure symmetry
	n_edges = {}
	filtered_edge_dict = {}
	for mol,edges in edge_dict.items():
		new_edges = []
		remove_index = []
		for i,edge in enumerate(edges):
			mol2,score = edge
			targetmols,_ = zip(*edge_dict[mol2])
			if mol in targetmols:
				new_edges.append(edge)
		filtered_edge_dict[mol] = new_edges
		
		n_edges[mol] = len(new_edges)
	return filtered_edge_dict,n_edges

def make_network(edge_dict,max_nodes = 100):
	components = {}
	# remove the singletons
	sing_id = -1
	done = set()
	component_index = 0
	for mol,edges in edge_dict.items():
		if len(edges) == 0:
			components[mol] = sing_id
			done.add(mol)
	component_thresholds = {}
	for mol in edge_dict:
		if mol in done:
			continue
		# build a component from this mol
		too_big = True
		min_score = 0.7
		while too_big:
			min_seen = 1.0
			finished = False
			search_pos = 0
			members = [mol]
			done.add(mol)
			while not finished:
				c = members[search_pos]
				target_scores = edge_dict[c]
				for t,s in target_scores:
					if (not t in done) and (s > min_score):
						members.append(t)
						done.add(t)
						if s < min_seen:
							min_seen = s
				search_pos += 1
				if search_pos == len(members):
					finished = True
			n_members = len(members)
			# print n_members
			if n_members <= max_nodes:
				too_big = False
			else:
				min_score = min_seen
				done = done - set(members)
			# print min_seen
		print len(members)
		if n_members == 1:
			components[mol] = sing_id
		else:
			for mol2 in members:
				components[mol2] = component_index
			component_thresholds[component_index] = min_score
			component_index += 1

		


	return components,component_thresholds

