{% load staticfiles %}

#### During the use of Mass2Motifs for structural grouping, annotation, and/or classification, please keep the following in mind:

---

###### Mass2Motifs are collections of co-occurring mass fragments and/or neutral losses

Mass2Motifs do not represent *real fragmentation spectra* – they are groups of mass fragments and/or neutral losses that often co-occur in a large collection of fragmentation spectra, i.e., from a set of reference compounds or from a LC-MS/MS run. Whilst we can *reconstruct* those collections in a spectral format, these are ‘fragmentation spectra’ without precursor mass and if present neutral losses usually do not ‘match’ to a Mass2Motif fragment unlike in real fragmentation spectra. 

---

###### Mass2Motifs can be indicative for the presence of a substructure that is not fully reflected in the motif

A Mass2Motif can encompass a complete substructure or *indicate* the presence of a substructure (building block of molecule) or structural feature (small generic (functional) group like hydroxyl or carboxylic acid group) rather than fully encompassing it by fragments and losses. To illustrate this, take for example the Adenine Mass2Motif that contains fragments and losses including the fragment peak that is characteristic for the complete adenine molecule (C5H6N5 – [M+H]+) and fragments thereof. In contrast, in (naturally amino-acid rich) beer and urine samples, the Mass2Motif *indicative* for Phenylalanine consists of several fragments that encompass the [Phenylalanine-CHOOH] substructure rather than the complete Phenylalanine structure. However, presence of the Mass2Motif is still indicative for the presence of Phenylalanine in beer and urine, especially in combination with metadata about the sample from which the fragmentation spectra were obtained (i.e., biological origin etc.).

---

###### Mass2Motifs for the same substructure can differ when discovered from different data sets

In practice, the exact Mass2Motif composition of fragments/losses characteristic for a specific substructure can differ slightly when learnt from different input data sets – influenced by the instrument, fragmentation, and/or sample type. However, when comparing Mass2Motifs using cosine scoring one will find scores higher than 0.75 for many of such cases – making Mass2Motif matching from other datasets an efficient way of initial Mass2Motif characterization.

---

###### Mass2Motifs can encompass isomeric substructures 

Mass spectrometry is sometimes – but not always – able to differentiate between *positional* isomers, and seldom able to differentiate between *stereo* isomers. Likewise, Mass2Motifs could point to several *isomeric* substructures. Again, with help of metadata one could in quite a few cases rationalize which is the most likely candidate substructure. 