MS1 Analysis for Decomposition - Frank 21st April 2017
==

When experiment type has been set to *Decomposition*, you can upload .mzML and .csv file and use exisiting Mass2Motif sets to do decomposition. The peaklist file (.csv) will be parsed, and stored in *Sample* and *DocSampleIntensity* tables.

Since each experiment can have multiple decompositions, for MS1 analysis, *DecompositionAnalysis*, *DecompositionAnalysisResult*, and *DecompositionAnalysisResultPlage* has been created to store analysis data and results.

Decomposition used an extra layer of abstraction *GlobalMotif* to keep things clean. 

A new function *get_docglobalm2m* has been created based on *get_docm2m*, and use *GlobalMotif* here instead of *Mass2Motif* in LDA experiment.

*make_decomposition_graph* and *get_graph* are similar to the ones in basicviz/views/views_lda_single.py for LDA.

*create_ms1analysis_decomposition* and *process_ms1_analysis_decomposition* functions have been created for processing MS1 analysis for decomposition.

Other settings are same to original MS1 analysis.