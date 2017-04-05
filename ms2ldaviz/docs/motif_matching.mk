Motif Matching - Instructions and details, SR 5th April 2017
==

# Introduction

When a new LDA analysis is run, annotating the motifs is a tedious process. In many cases we will be re-discovering motifs that have previously been annotated. In such cases motif matching allows the user to compare some learnt motifs against those from a previous experiment. If matches are found, motifs can be linked to one another so that annotations from one (the *linked to* motif are visible from the *linked* motif).

# Instructions

Two new menu options have been provided (currently only for `extra users`). To get started, click on Start Motif Matching from the experiment you would like to link motifs to. i.e. you have just done experiment A and want to see if any motifs you previously annotated in B are present, so you click Start motif matching from experiment A.

You will see a form with two fields. In the first, choose the experiment you want to compare against. In the second choose the minimum cosine similarity to temporarily store the link (i.e. it does not formally link motifs at this stage, but just finds the matches and stores them if they exceed this threshold).

When you click Submit the job will start and you will be re-directed to your home page.

When the job has finished (no nice notifications yet) you can look at the results by clicking on the Manage motif matches option in the menu (again for, say experiment A). You will see a list of the matches along with various bits of information about the motifs in question and links so that you can look at them more closely. On the right hand side of the table, you will see two actions: add link and remove link. The first formally adds the link to the *linked to* motif (i.e. the motif in experiment A). The second removes any link associated with the *linked to* motif. Note that as soon as you link, the annotations belonging to the motif in experiment B will be visible from the motif in A.

# Annotation behaviour

Note that there are now two ways in which a motif can be annotated. By manually entering an annotation (as previously) or by linking to another motif. The former takes precedent. So, if an annotation has been manually added, it will appear ahead of any link annotations. This is the case if the manual annotation is added *before* or *after* the link.

# Details

A couple of important points in the implementation. To compute the similarities, we need both motifs in the same feature space. To do this, the method initially matches features. It first tries to match on the feature name (this will work well if binned features are used). If, for a feature in experiment A it can't find a match this way, it looks to see if the feature's mz falls between the `min_mz` and `max_mz` attributes of any of the features in experiment B.

Once in the same feature space, the cosine similarity can be computed. Note that the vector norms used to normalise the cosine score are computed in the original space (i.e. we don't just use the features that can be matched) to ensure we don't get erroneously high scores when, say, only one feature is mapped.