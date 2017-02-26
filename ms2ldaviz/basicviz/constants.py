AVAILABLE_OPTIONS = [('doc_m2m_threshold', 'Probability threshold for showing document to mass2motif links'),
                     ('log_peakset_intensities', 'Whether or not to log the peakset intensities (true,false)'),
                     ('peakset_matching_tolerance', 'Tolerance to use when matching peaksets'),
                     ('heatmap_minimum_display_count',
                      'Minimum number of instances in a peakset to display it in the heatmap'),
                     ('default_doc_m2m_score',
                      'Default score to use when extracting document <-> mass2motif matches. Use either "probability" or "overlap_score", or "both"'),
                     ('heatmap_normalisation','how to normalise the rows in the heatmap: none, standard, max')]

EXPERIMENT_STATUS_CODE = [
    ('0', 'Pending'),
    ('1', 'Ready'),
]

EXPERIMENT_TYPE = [
    ('0', 'LDA'),
    ('1', 'Decomposition'),
]

EXPERIMENT_DECOMPOSITION_SOURCE = [
    ('N', 'No'),
    ('Y', 'Yes'),
]

MASSBANK_AUTHORS = "van der Hooft, J. J. J., Wandy J., Rogers, S., University of Glasgow"
MASSBANK_LICENSE = 'CC BY'
MASSBANK_PUBLICATION = "van der Hooft, J. J. J., Wandy, J., Barrett, M, P., Burgess, K. E. V. & Rogers, S. (2016). Topic modeling for untargeted substructure exploration in metabolomics. Proceedings of the National Academy of Sciences. 113(48), 13738-13743. http://doi.org/10.1073/pnas.1608041113"
MASSBANK_COPYRIGHT = "Copyright (C) 2016 University of Glasgow"
MASSBANK_AC_INSTRUMENT = 'Q Exactive Orbitrap Thermo Fisher Scientific'
MASSBANK_AC_INSTRUMENT_TYPE = 'ESI-QFT'
MASSBANK_IONISATION = 'ESI'
MASSBANK_MS_DATA_PROCESSING = 'WHOLE MS2LDA Analysis 1.0'