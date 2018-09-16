import pylab as plt
import numpy as np



def compute_motif_degrees(lda_dict,p_thresh,o_thresh):
	motifs = lda_dict['beta'].keys()
	motif_degrees = {m:0 for m in motifs}
	docs = lda_dict['theta'].keys()
	for doc in docs:
		for motif,p in lda_dict['theta'][doc].items():
			if p >= p_thresh:
				o = lda_dict['overlap_scores'][doc].get(motif,0.0)
				if o >= o_thresh:
					motif_degrees[motif] += 1
	md = zip(motif_degrees.keys(),motif_degrees.values())
	md.sort(key = lambda x: x[1],reverse = True)
	return motif_degrees,md


def plot_motif(lda_dict,motif_name,xlim=None,**kwargs):
	be = lda_dict['beta'][motif_name]
	plot_motif_from_dict(be,xlim=xlim,**kwargs)

def plot_motif_from_dict(motif_dict,xlim=None,**kwargs):
	plt.figure(**kwargs)
	anylosses = False
	for feature,intensity in motif_dict.items():
		if feature.startswith('fragment'):
			# deal with losses later
			feature_mz = float(feature.split('_')[1])
			plt.plot([feature_mz,feature_mz],[0,intensity],'r')
		elif feature.startswith('loss'):
			feature_mz = float(feature.split('_')[1])
			plt.plot([feature_mz,feature_mz],[0,-intensity],'b')			
			anylosses = True

	if xlim:
		plt.xlim(xlim)
	if anylosses:
		plt.plot(plt.xlim(),[0,0],'k--')
	plt.xlabel('m/z')
	plt.ylabel('probability')



def list_metadata_fields(lda_dict):
	fields = []
	for doc,md in lda_dict['doc_metadata'].items():
		these_keys = md.keys()
		fields += these_keys
		fields = list(set(fields))
	return fields

def print_mols(lda_dict,mols,fields = ['precursormass','parentrt']):
	heads = "{:20s}\t".format("Spectra name")
	for field in fields:
		heads += "{:20s}\t".format(field)
	print heads
	for doc in mols:
		md = lda_dict['doc_metadata'][doc]
		line = "{:20s}\t".format(doc)
		for field in fields:
			line += "{:20s}\t".format(str(md.get(field,'NA')))
		print line

def get_motif_mols(lda_dict,motif,p_thresh,o_thresh):
	theta = lda_dict['theta']
	mols = []
	for mol,motifs in theta.items():
		p = motifs.get(motif,0.0)
		o = lda_dict['overlap_scores'][mol].get(motif,0.0)
		if p >= p_thresh and o >= o_thresh:
			mols.append(mol)
	return mols

def plot_mol(lda_dict,mol,color_motifs = False,xlim = None,**kwargs):
	plt.figure(**kwargs)
	spec = lda_dict['corpus'][mol]

	motif_probs = lda_dict['theta'][mol]
	motif_probs = zip(motif_probs.keys(),motif_probs.values())
	motif_probs.sort(key = lambda x: x[1],reverse = True)

	n_motifs = min(len(motif_probs),4)
	top_motifs = [m for m,p in motif_probs[:n_motifs]]
	cols = ['r','g','b','k']
	motif_cols = {m:cols[i] for i,m in enumerate(top_motifs)}
	if not color_motifs:
		for feature,intensity in spec.items():
			if feature.startswith('fragment'):
				mz = float(feature.split('_')[1])
				plt.plot([mz,mz],[0,intensity],'r')
			elif feature.startswith('loss'):
				mz = float(feature.split('_')[1])
				plt.plot([mz,mz],[0,-intensity],'b')

	else:
		anylosses = False
		phi = lda_dict['phi'][mol]
		for feature,pphi in phi.items():
			if feature.startswith('fragment'):
				mul = 1
			elif feature.startswith('loss'):
				mul = -1
				anylosses = True
			else:
				continue

			mz = float(feature.split('_')[1])
			total_intensity = spec[feature]
			cum_intensity = 0
			col_index = 0
			for motif,prob in pphi.items():
				plt.plot([mz,mz],[mul*cum_intensity,mul*(cum_intensity + prob*total_intensity)],color=motif_cols.get(motif,[0.6,0.6,0.6]))
				cum_intensity += prob*total_intensity
				col_index += 1
				if col_index == len(cols):
					break
			if cum_intensity < total_intensity:
				plt.plot([mz,mz],[mul*cum_intensity,mul*total_intensity],'k',color = [0.6,0.6,0.6])
		tit_string = "    ".join(["{}: {}".format(m,c) for m,c in motif_cols.items()])
		plt.title(tit_string)
		if anylosses:
			plt.plot(plt.xlim(),[0,0],'k--')
	if xlim:
		plt.xlim(xlim)


def match_motifs(lda_dict,db_motifs,threshold = 0.7):
	matches = []
	for motif,spec in lda_dict['beta'].items():
		for db_motif,db_spec in db_motifs.items():
			score = compute_similarity(spec,db_spec)
			if score > threshold:
				matches.append((motif,db_motif,score))
	matches.sort(key = lambda x: x[2], reverse = True)
	return matches

def compute_similarity(spec1,spec2):
    # compute the cosine similarity of the two spectra
    prod = 0.0
    i1 = 0.0
    for mz,intensity in spec1.items():
        i1 += intensity**2
        intensity2 = spec2.get(mz,0.0)
        prod += intensity * intensity2
    i2 = sum([i**2 for i in spec2.values()])
    if i2 == 0:
    	return 0.0
    sim = prod/(np.sqrt(i1)*np.sqrt(i2))
    return sim