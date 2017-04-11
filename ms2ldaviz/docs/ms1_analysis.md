MS1 Analysis - Frank 10th April 2017
==

# Introduction

When a new LDA experiment is created, MS1(peaklist) and MS2 files are uploaded. Currently, peaklist processing for MS1 analysis has only been implemented in LoadMZML class. For differential expression analysis, intensities of different samples can be stored in peaklist file (.csv). The intensity for each document and each sample will be stored in database. When doing MS1 analysis, users can choose sample names as *group1* and *group2* and can give the choice of using logarithm or not for intensity data.

# Input

with following format:
- header line: "...,mass,RT,samplename_1, samplename_2,..."
- delimiter: ','
- *mass* can have col name: mass, mz (case-insensitive)
- *retention time* can have col name: rt (case-insensitive)
- columns before *mass* will be ignored
- missing data can be stored as empty, or NA in .csv file. After line spliting, will use *try except* to judge if the intensity is acceptable. Empty, string format and negative intensities will not be stored in database.

# Database

Four tables have been created for MS1 analysis:
- Sample
- DocSampleIntensity
- Analysis
- AnalysisResult

*Sample* only stores sample names, *DocSampleIntensity* stores intensity for each document and sample, *Analysis* stores MS1 analysis settings like choose which samples as group1 and group2, *AnalysisResult* stores results of each analysis, including *fold change* and *pValue*.

# Visualization

document size: min(5 - np.log(pVlaue) * 15, 50)
If pValue is nan after ttest_ind calculation, set size to be 1 (use smallest size).

document colour: use np.log(fold change) to represent
- min_logfc: blue, [0,0,255]
- max_logfc: red, [255,0,0]
- logfc=0: white, [255,255,255]
If intensity is missing, ignore it when calculating fold change. If other exception happens, set fold change to be 1 (use white colour). 

Exception:
Intensities between 0 and 1 (in log and non-log mode) are very rare. If this happens, set the document colour to be white, otherwise other documents' colour will be skewed by colour scaling.

# celery

After MS1 analysis options have been set, the calculation of fold change and pValue and database insertion will be run in celery.

# logarithm

Give user choice from log and non-log of intensity data in create analysis page. 
