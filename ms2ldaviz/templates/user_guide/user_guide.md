{% load staticfiles %}

### 1. Logging in to Ms2lda.org:

To log in to Ms2lda.org, go to the <a href="http://ms2lda.org/registration/login/" target="_blank" title="Ms2lda">Login</a> page.

Once there, please input your username and password. If you have not been provided with a username and password, 
please email us to create an account. Once you hit enter, or click 'Submit', you should find yourself in your 
main page listing all the experiments that you have access to (whether in edit or read-only modes). 

There are two experiment types that can be created on Ms2lda.org:

- LDA experiments have Mass2Motifs (patterns of co-occuring fragment and neutral loss features) 
that potentially indicate structural families to be inferred in a completely unsupervised manner from the data using
LDA
- Decomposition experiments used pre-defined Mass2Motifs that could be annotated from another experiment
in your data.

---

### 2. Prerequisites

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

### 3. Analysing Your Data

Once processed, LDA experiments (where substructures are discovered in an unsupervised manner) can be found in the list of LDA experiments.
Experiments that are editable are shown in **bold** in the list, while read-only experiments are shown in normal font weight.
Clicking an experiment will expand it into tabs, where additional functionalities can be selected.

<!-- ![LDA Experiments][lda_experiment] -->
<p class="centered">
<img src="{% static 'images/user_guide/lda_experiments.png' %}" class="img-thumbnail" alt="LDA Experiments" width="100%" style="float:none;margin:auto;">
</p>

The following functionalities are available for an LDA experiment:

- **Summary Page**: displays a summary of key results in the data, including extracted fragment and loss features, inferred Mass2Motifs and spectra that can be explained by these motifs. This should be your first place of exploring MS2LDA results.
- **Show Fragmentation Spectra**:  displays the fragmentation spectra in the data, alongside Mass2Motifs that explain the features in those spectra.
- **Show Mass2Motifs**: diplays inferred Mass2Motifs.
- **View Experiment Options**: displays a list of configurable options when performing motif matching and visualisating the data.
- **Create MS1 Analysis**: if MS1 data is provided, this allows you to create MS1 analysis of case vs. control study where the differential prevalence of Mass2Motifs can be compared and visualised.
- **Start Motif Matching**: performs the matching of inferred Mass2Motifs for this data against Mass2Motifs from another data. This allows for quick annotations. 
- **Manage Motif Matches**: once started, motif matching will run in the background. The results will be shown here and can be updated.
- **Start Visualisation**: allows you to visualise Mass2Motifs and fragmentation spectra in a network graph.

The following sections describe all the functionalities of Ms2lda.org in greater details:

---

#### 4. Creating an Experiment in Ms2lda.org

To create your experiment, first give it an experiment name and a description. Choose an experiment type (either **LDA** or **Decomposition**), and select the format of the MS2 fragmentation file (either .mzML or .MSP or .MGF). Finally, upload the fragmentation file in the correct format in the file selector. 

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

#### 5. Summary Page

The **Summary Page** shows all the key results for your dataset. In particular, inferred Mass2Motifs can be annotated from the Summary Page by clicking Mass2Motif in the **Mass2Motif Details** table of the Summary Page. In the next screen that appears, details on the selected Mass2Motif is shown. Annotation can be assigned from this screen. In the example below, we assign the annotation "Histidine substructure" to this Mass2Motifs based on the top fragments (110.07176, 156.07684, etc) shown in the table.

<!-- ![Motif Annotation][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_annotation.png' %}" class="img-thumbnail" alt="Motif Annotation" width="100%" style="float:none;margin:auto;">
</p>

---

#### 6. Motif Matching

For quick annotations of a large number of Mass2Motifs, manual annotating can be tedious. The motif matching functionality can be used to speed up this process. This functionality is launched from the **Start Motif Matching** link from the functionality tabs of an experiment. Matching is performed based on the cosine similarity, which is specified as a user-configurable option. To begin motif matching, select a motifset to match against and specify the minimum cosine similarity score to select candidate matches. Click the **Start matching** button.

<!-- ![Motif Matching (Start)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_matching_start.png' %}" class="img-thumbnail" alt="Motif Matching (Start)" width="100%" style="float:none;margin:auto;">
</p>

Matching will be performed in the background. Upon completion, match results will be shown in the **Manage Motif Matches** screen, as shown below. The first column shows the original Mass2Motifs inferred in this dataset. The second and third columns show the best match Mass2Motifs (according to cosine similarity) in the target dataset. The match score is shown in the next column. Clicking **Add Link** will create a link between the pair of Mass2Motifs, transferring their annotations from the matched to the original Mass2Motif.

<!-- ![Motif Matching (Manage)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/motif_matching_manage.png' %}" class="img-thumbnail" alt="Motif Matching (Manage)" width="100%" style="float:none;margin:auto;">
</p>

---

#### 7. Visualisation

Interactive visualisation can be launched from the **Start Visualisation** link from the functionality tabs of an experiment. The minimum degree is the minimum threshold to set to draw an edge connecting a Mass2Motifs to adjacent spectra that can be explained by that Mass2Motif, e.g. a value of 5 means edges are drawn only when a Mass2Motif is connected to 5 spectra (at the specified threshold in the experiment option).

<!-- ![Visualisation (Start)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/visualisation_start.png' %}" class="img-thumbnail" alt="Visualisation (Start)" width="100%" style="float:none;margin:auto;">
</p>

The next screen shows the interactive visualisation. Circles are Mass2Motifs, while squares are spectra. Selecting a Mass2Motif in the network will display more information in other panels, including the fragmentation spectra that are explained by this motif and the counts of occurrences of this motif amongst the spectra. Motif nodes in the network can also be searched through the search box at the top of the page.

<!-- ![Visualisation (Network)][motif_annotation] -->
<p class="centered">
<img src="{% static 'images/user_guide/visualisation_network.png' %}" class="img-thumbnail" alt="Visualisation (Network)" width="100%" style="float:none;margin:auto;">
</p>

---

#### 8. Decomposition

Decomposition allows for spectra in a dataset to be decomposed onto a set of pre-defined Mass2Motifs from another experiment. There are two key advantages of performing decomposition: it is faster than normal LDA because Mass2Motifs do not have to be inferred again, and also annotations can be easily transferred from the original pre-defined Mass2Motifs to new and unseen dataset (without having to perform another motif matching step). Decomposition experiments can be created the **Create Experiment** page and selecting **Decomposition** as the experiment type.

---

#### 9. MS1 Analysis

Where available, MS1 analysis can be performed to infer the differential expression of Mass2Motifs (potential substructures) across case and control samples.

#### 10. Guest Data

To log into MS2lda.org, you need to create an account. However, a guest account is also available to explore the system without registration. Limited functionalities will be available.