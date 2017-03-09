# scipt to make combined motifset

combined_filename = '/home/combined_motifs.csv'
import csv
motifs = []
with open(combined_filename,'r') as f:
	reader = csv.reader(f)
	for line in reader:
		motifs.append(line)