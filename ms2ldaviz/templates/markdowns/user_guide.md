{% load static %}

## Table of Contents

1. <a href="#getting_started">Getting Started</a>
2. <a href="#prerequisites">Prerequisites</a>
3. <a href="#analysing_your_data">Analysing Your Data</a>
4. <a href="#creating_an_experiment">Creating an Experiment</a>
5. <a href="#summary_page">Summary Page</a>
6. <a href="#mass2motif_matching">Mass2Motif Matching</a>
7. <a href="#visualisation">Interactive Network Visualisation</a>
8. <a href="#decomposition">Decomposition</a>
9. <a href="#quick_decomposition">Quick Decomposition of One MS/MS spectra</a>
10. <a href="#ms1_analysis">Combining MS1 Differential Expression or Prevalence with MS2LDA</a>
11. <a href="#guest_data">Guest Data</a>

---

#### <a name="getting_started">1. Getting Started</a>

To log in to Ms2lda.org, go to the <a href="http://ms2lda.org/registration/login/" target="_blank" title="Ms2lda">Login</a> page.

Once there, please input your username and password. If you have not been provided with a username and password, 
please email us to create an account. Once you hit enter, or click 'Submit', you should find yourself in your 
main page listing all the experiments that you have access to (whether in edit or read-only modes). 

There are two experiment types that can be created on Ms2lda.org:

- LDA experiments have Mass2Motifs (patterns of co-occuring fragment and neutral loss features) 
that potentially indicate structural families to be discovered in a completely unsupervised manner from the data using LDA

- Decomposition experiments used pre-defined Mass2Motifs that could be annotated from another experiment
in your data.

---

#### <a name="prerequisites">2. Prerequisites</a>

To analyse your data in Ms2lda.org, you first need:

1. Your fragmentation data in mzML, MSP or MGF formats
2. A list of MS1 peaks (optional). When fragmentation data in mzML format is provided, this list will be used to seed the MS1 peaks during
feature extraction so only peaks that match the MS1 list within certain m/z and RT tolerances will be used.

Once you have these available, from the Experiment screen, click on the **Create Experiment** button, shown in <strong><font color="red">(A)</font></strong> below. 
A screen will appear asking you to upload your data and define the parameters for feature extraction and inference (see Section 4 for more details).
Upon clicking submit, the experiment will be processed in a job queue. While processing, it is also shown in the list of **Pending Experiments**, shown in <strong><font color="red">(B)</font></strong> below. 
Upon completion, experiments are moved to the list of LDA or Decomposition experiments, depending on the experiment type that you have specified.

<!-- ![Create Experiment][create_experiment] -->
<p class="centered">
<img src="{% static 'images/user_guide/create_experiment.png' %}" class="img-thumbnail" alt="Create Experiment" width="100%" style="float:none;margin:auto;">
</p>

---

#### <a name="#analysing_your_data">3. Analysing Your Data</a>

Once processed, LDA experiments (where substructures are discovered in an unsupervised manner) can be found in the list of LDA experiments.
Experiments that are editable are shown in **bold** in the list, while read-only experiments are shown in normal font weight.
Clicking an experiment will expand it into tabs, where additional functionalities can be selected.

<!-- ![LDA Experiments][lda_experiment] -->
<p class="centered">
<img src="{% static 'images/user_guide/lda_experiments.png' %}" class="img-thumbnail" alt="LDA Experiments" width="100%" style="float:none;margin:auto;">
</p>

The following functionalities are available for an LDA experiment:

- **Summary Page**: displays a summary of key results in the data, including extracted fragment and loss features, discovered Mass2Motifs and spectra that can be explained by these motifs. This is advised to be your first place of exploring MS2LDA results.
- **Show Fragmentation Spectra**:  displays the fragmentation spectra in the data, alongside Mass2Motifs that explain the features in those spectra.
- **Show Mass2Motifs**: diplays discovered Mass2Motifs.
- **View Experiment Options**: displays a list of configurable options when performing motif matching and visualisating the data.
- **Create MS1 Analysis**: if MS1 data is provided, this allows you to create MS1 analysis of case vs. control study where the differential prevalence of Mass2Motifs can be compared and visualised.
- **Start Motif Matching**: performs the matching of discovered Mass2Motifs for this data against Mass2Motifs from another data. This allows for quick annotations. 
- **Manage Motif Matches**: once started, motif matching will run in the background. The results will be shown here and can be updated.
- **Start Visualisation**: allows you to visualise Mass2Motifs and fragmentation spectra in a network graph.

Most of these pages including the visualisation have an excellent **Search Function** where Mass2Motifs, Mass2Motif annotations, and/or parent ions can be quickly and convienently found. 

The following sections describe all the functionalities of Ms2lda.org in greater details:

---

#### <a name="creating_an_experiment">4. Creating an Experiment</a>

To create your experiment, click on **Create Experiment**. First give it an experiment name and a description. Choose an experiment type (either **LDA** or **Decomposition**), and select the format of the MS2 fragmentation file (either .mzML or .MSP or .MGF). Finally, upload the fragmentation file in the correct format in the file selector. 

<!-- ![Create Experiment][create_experiment_top] -->
<p class="centered">
<img src="{% static 'images/user_guide/create_experiment_top.png' %}" class="img-thumbnail" alt="Create Experiment" width="100%" style="float:none;margin:auto;">
</p>

Depending on your choice of fragmentation file format, different fields will be shown to configure filtering and feature extraction parameters. For fragmentation data in .mzML format, the parameters are:

- **Fragmentation isolation window** should be adjusted based on your instrument settings.
- **Mass and retention time tolerances when linking peaks** are only used when the MS1 peaklist is provided. The default values should work reasonably well.
- **Minimum and maximum retention time and intensity of MS1 peaks** are used to filter the data. Make sure to adjust them based on your instrument settings.
- **Attempt to filter out duplicate MS1 peaks** are used to merge peaks within the specified mass and RT tolerance windows.

<!-- ![Create Experiment (mzML)][create_experiment_mzml] -->
<p class="centered">
<img src="{% static 'images/user_guide/create_experiment_mzml.png' %}" class="img-thumbnail" alt="Create Experiment (mzML)" width="100%" style="float:none;margin:auto;">
</p>

For fragmentation data in .MSP or .MGF format, the parameters are:

- **Minimum intensity of MS1 peaks to store** is used to filter MS1 peaks by the specified minimum intensity value.
- **Minimum intensity of MS2 peaks to store** is used to filter MS2 peaks by the specified minimum intensity value.

<!-- ![Create Experiment (MSP, MGF)][create_experiment_msp_mgf] -->
<p class="centered">
<img src="{% static 'images/user_guide/create_experiment_msp_mgf.png' %}" class="img-thumbnail" alt="Create Experiment (MSP, MGF)" width="100%" style="float:none;margin:auto;">
</p>

For all formats, the following parameters for LDA inference have to be specified:

- **Number of Mass2Motifs**: specifies the number of Mass2Motifs (topics) to infer in the LDA alrogithm. Based on our experience, a value between 300 - 500 are usually sufficient.
- **Number of iterations (for LDA)**:  specifies the number of steps to perform during variational inference. The default value of 1000 is usually enough.

Finally press the **Submit Your Experiment** button to submit the experiment.

---

#### <a name="summary_page">5. Summary Page</a>

The **Summary Page** shows all the key results for your dataset. In particular, the discovered Mass2Motifs can be studied and annotated from the Summary Page by clicking Mass2Motif in the **Mass2Motif Details** table of the Summary Page. The Table contains the degrees and annotations (if there). When clicking on a Mass2Motif link, details on the selected Mass2Motif are shown. Annotation can also be assigned from this screen. In the example below, we assign the annotation "Histidine substructure" to this Mass2Motifs based on the top fragments (110.07176, 156.07684, etc) shown in the table. The Mass2Motifs can also be assessed through the **Show Mass2Motifs Page**.

<!-- ![Motif Annotation][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_annotation.png' %}" class="img-thumbnail" alt="Motif Annotation" width="100%" style="float:none;margin:auto;">
</p>

---

#### <a name="mass2motif_matching">6. Mass2Motif Matching</a>

For quick annotations of a large number of Mass2Motifs, manual annotation can be tedious. The motif matching functionality can be used to speed up this process. This functionality is launched from the **Start Motif Matching** link from the functionality tabs of an experiment. Matching is performed based on the cosine similarity, which is specified as a user-configurable option. To begin motif matching, select a motifset to match against and specify the minimum cosine similarity score to select candidate matches. Click the **Start matching** button.

<!-- ![Motif Matching (Start)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_matching_start.png' %}" class="img-thumbnail" alt="Motif Matching (Start)" width="100%" style="float:none;margin:auto;">
</p>

Matching will be performed in the background. Upon completion, match results will be shown in the **Manage Motif Matches** screen, as shown below. The first column shows the original Mass2Motifs discovered in this dataset. The second and third columns show the best match Mass2Motifs (according to cosine similarity) in the target dataset. The match score is shown in the next column. Clicking **Add Link** will create a link between the pair of Mass2Motifs, transferring their annotations from the matched to the original Mass2Motif. It is important to realize that if the matched annotation changes, so will the annotation of the linked Mass2Motif.

<!-- ![Motif Matching (Manage)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_matching_manage.png' %}" class="img-thumbnail" alt="Motif Matching (Manage)" width="100%" style="float:none;margin:auto;">
</p>

---

#### <a name="visualisation">7. Interactive Network Visualisation</a>

Interactive visualisation can be launched from the **Start Visualisation** link from the functionality tabs of an experiment. The minimum degree is the minimum threshold to set to draw an edge connecting a Mass2Motifs to adjacent spectra that can be explained by that Mass2Motif, e.g. a value of 5 means edges are drawn only when a Mass2Motif is connected to 5 spectra (at the specified threshold in the experiment option). Please note that if all Mass2Motifs contain more than 5 spectra, the network might take a while to load and we advise users to higher the minimum degree for interactive network visualisation. 

<!-- ![Visualisation (Start)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/visualisation_start.png' %}" class="img-thumbnail" alt="Visualisation (Start)" width="100%" style="float:none;margin:auto;">
</p>

The next screen shows the interactive visualisation. Circles are Mass2Motifs, while squares are spectra (fragmented metabolites). If the network appears as a small pile of circles, please use the left-click mouse to select a Mass2Motif and drag it slightly away from the pile - the network will 'explode' as result. The network can be enlarged or made smaller by zooming in or out using the mouse wheel or a similar action. Selecting (double-clicking on) a Mass2Motif in the network will display more information in other panels, including the fragmentation spectra that are explained by this motif and the counts of occurrences of this motif amongst the spectra. Associated spectra (fragmented metabolites) will be highlighted as well after selecting the Mass2Motif. Annotated Mass2Motifs will be coloured red in the network and the annotations will be visible when hoovering over them with the mouse. Other Mass2Motifs will appear orange and Mass2Motif numbers will appear when hoovering over them with the mouse. Similarly, information on the fragmented metabolites including the precursor ion will appear when hoovering over the squares with the mouse. Motif nodes, annotations, and fragmented ions in the network can also be searched through the search box at the top of the page and subsequently quickly and convienently selected in the network.

<!-- ![Visualisation (Network)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/visualisation_network.png' %}" class="img-thumbnail" alt="Visualisation (Network)" width="100%" style="float:none;margin:auto;">
</p>

---

#### <a name="decomposition">8. Decomposition</a>

Decomposition allows for spectra in a dataset to be decomposed onto a set of pre-defined Mass2Motifs from another experiment. There are two key advantages of performing decomposition: firstly, it is faster than normal LDA because Mass2Motifs do not have to be defined again, and secondly, annotations are directly transferred from the original pre-defined Mass2Motifs to new and unseen dataset (without having to perform a motif matching step). However, it has to be taken into account that substructures not present in the pre-defined Mass2Motif set, they will not be covered or recognized by the decomposition experiment. Decomposition experiments can be created the **Create Experiment** page and selecting **Decomposition** as the experiment type. It is encouraged to take a pre-defined Mass2Motif set discovered in (i.e., inferred from) standards like the MassBank or GNPS set.

Similar functionalities are available to analyse the results with **Show Fragmentation Spectra**, **Show Mass2Motifs**, **Create MS1 analysis**, and **Start Visualisation** described above. The **Create New Decomposition** functionality allows for a quick way to start a new decomposition using the same fragmentation file but using a different pre-defined Mass2Motif set. 

---

#### <a name="quick_decomposition">9. Quick Decomposition of One MS/MS spectra</a>

After clicking on **<a href="http://ms2lda.org/decomposition/decompose_spectrum/">Quick Decomposition</a>**, one spectrum can be uploaded to the website. An example spectrum upload is shown. Please replace the parent ion and the framgents and intensities in the screen and select a Mass2Motif set. After submitting, a link will be visible where the results will be visible after the spectrum has been processed. An example result file is shown below:

> {"alpha": [], "motifset": "massbank_motifset", "decompositions": {"spectrum_188.0818": [["motif_275", "motif_275", 0.6771396143585594, 0.16086854487297184, null], ["motif_18", "motif_18", 0.21212121212121213, 0.9366374987030056, "CO loss - indicative for presence of ketone/aldehyde/lactone group (C=O)"], ["motif_164", "motif_164", 0.046389814537547945, 0.4116581824551805, null], ["motif_315", "motif_315", 0.02338064854620109, 0.6408839113935687, null], ["motif_202", "motif_202", 0.010499139755035958, 0.03257188481258688, null], ["motif_417", "motif_417", 0.006181302909985723, 0.01915199122760524, null], ["motif_461", "motif_461", 0.00535089789408206, 0.027940452512704973, null], ["motif_250", "motif_250", 0.005285733171085548, 0.07246537561256258, null], ["motif_30", "motif_30", 0.0041592394533571005, 1.0, "Fragment indicative for aromatic compounds related to methylbenzene substructure (C7H7 fragment)"]]}}   

---

#### <a name="ms1_analysis">10. Combining MS1 Differential Expression or Prevalence with MS2LDA</a>

Where available, MS1 analysis can be performed to map differential expression or prevalence of metabolites based on their MS1 intensities. In order to do so, you will need to upload a CSV file that includes the fragmented sample. Please look carefully at the requirements for the CSV file at the bottom of the **Create Experiment** page. Please also note that this is currently possible alongside an mzml file only. During the preprocessing, the fragmented features of the fragmentation mzml file will be matched to those present in the CSV file based on their m/z values and retention times, so ensure that the retention times are comparable and that the m/z and retention time matching parameters are set correctly. Once the MS1 features are matched, MS1 analysis can be done using the **Create MS1 analysis**. The list of sample names is available in the middle of the page (see figure below), and after selection of samples (e.g., 5 treatment replicates of which one was fragmented) the arrows can be used to move the samples to group 1 on the left. Similarly, group 2 (on the right) can be populated. A t-test comparative analysis is performed with group 1 over group 2.

<!-- ![Differential Expression][differential_expression] -->
<p class="centered">
<img src="{% static 'images/user_guide/MS1analysis_userguideline_image_2_network.PNG' %}" class="img-thumbnail" alt="Differential Expression" width="100%" style="float:none;margin:auto;">
</p>

In order to analyze the MS1 analysis, it needs to be mapped on the network in the visualisation page. After loading the network in the visualisation page, the user can toggle **Show MS1 analysis in the network** at the left bottom of the page which will change the appearance of the network (see Figure below). Now, Mass2Motifs are coloured green - the greener they are, the more differential metabolites contain that particular Mass2Motif. Additionally, differential metabolites are coloured (red is up, blue is down - the darker, the larger the fold change) and sized according to their significance, the larger the significance. As in the regular network, users can click on Mass2Motifs to view the spectra and other statistics. However, the Mass2Motif and/or number is now accompagnied by the PLAGE score (the higher, in the more differential metabolites the Mass2Motif is present, independent on the direction of the fold change). The user can return to the 'standard network view' by detoggling the **Show MS1 analysis in the network** option. 

<!-- ![Differential Expression (Network)][differential_expression_network] -->
<p class="centered">
<img src="{% static 'images/user_guide/MS1analysis_userguideline_image.PNG' %}" class="img-thumbnail" alt="Differential Expression (Network)" width="100%" style="float:none;margin:auto;">
</p>
  

---

#### <a name="guest_data">11. Guest Data</a>

To log into MS2lda.org, you need to create an account. However, a guest account is also available to explore the system without registration. Functionalities are available for most experiments to allow for browsing through example data sets. However, to create your own experiment, you will need to request an account to ensure that your data is visible to you and collaborators of your choice.

The following experiments are available for browsing:

- **LDA experiments**
-- gnps_binned_005: Mass2Motifs discovered from MS/MS spectra from 5770 GNPS standards (positive ionization mode) with many annotated Mass2Motifs. Guest users have viewing access to this.
-- massbank_binned_005: Mass2Motifs discovered from MS/MS spectra from 2132 MassBank standards (positive ionization mode) including annotated Mass2Motifs.
-- Beer6_POS_IPA_MS1_comparisons: Mass2Motifs discovered from MS/MS spectra from an IPA Beer (positive ionization mode). An MS1 analysis comparing IPA beers versus non-IPA beers is available for visualization. 

- **Decomposition experiment**
-- Beer3_POS_Decomposition_MassBankGNPScombinedset_MS1peaklist_MS1duplicatefilter: Seven Giraffes Extraordinary IPA beer decomposed using a combined MassBank and GNPS Mass2Motif set. As a consequence, many of the Mass2Motifs have annotations.
  
