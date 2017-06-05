MS1 Analysis - Frank 11th April 2017
==

# Introduction

When a new LDA experiment is created, MS1(peaklist) and MS2 files are uploaded. Currently, peaklist processing for MS1 analysis has only been implemented in LoadMZML class. For differential expression analysis, intensities of different samples can be stored in peaklist file (.csv). The intensity for each document and each sample will be stored in database. When doing MS1 analysis, users can choose sample names as *group1* and *group2* and can give the choice of using logarithm or not for intensity data.

# Input

with following format:
- header line: "...,mass,RT,samplename_1, samplename_2,..."
- delimiter: ','
- *mass* can only have col name: mass, mz (case-insensitive)
- *retention time* is assumed to be the next column immediately after *mass*
- columns before *mass* will be ignored
- missing data can be stored as empty, or NA in .csv file. After line spliting, will use *try except* to judge if the intensity is acceptable. Empty, string format and negative intensities will not be stored in database.

