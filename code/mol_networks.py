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
