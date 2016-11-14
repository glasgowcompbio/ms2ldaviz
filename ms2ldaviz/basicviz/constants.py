AVAILABLE_OPTIONS = [('doc_m2m_threshold', 'Probability threshold for showing document to mass2motif links'),
                     ('log_peakset_intensities', 'Whether or not to log the peakset intensities (true,false)'),
                     ('peakset_matching_tolerance', 'Tolerance to use when matching peaksets'),
                     ('heatmap_minimum_display_count',
                      'Minimum number of instances in a peakset to display it in the heatmap'),
                     ('default_doc_m2m_score',
                      'Default score to use when extracting document <-> mass2motif matches. Use either "probability" or "overlap_score", or "both"'),
                     ('heatmap_normalisation','how to normalise the rows in the heatmap: none, standard, max')]

DEFAULT_MASSBANK_AUTHORS = "van der Hooft, J. J. J., Wandy J., Rogers, S., University of Glasgow"
DEFAULT_MASSBANK_SPLASH = 'http://splash.fiehnlab.ucdavis.edu/splash/it'
DEFAULT_AC_INSTRUMENT = 'Q-Exactive (Thermo Fisher Scientific)'
DEFAULT_AC_INSTRUMENT_TYPE = 'LC-ESI-Orbitrap-MS'
DEFAULT_LICENSE = 'CC BY-SA'
DEFAULT_IONISATION = 'ESI'
