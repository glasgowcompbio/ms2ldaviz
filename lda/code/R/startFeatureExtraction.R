library(xcms)       # Load XCMS
library(RMassBank)  # Load RMassBank
library(gtools)     # Used for natural sorting
library(yaml)       # Used for reading configuration file
library(tools)

start_feature_extraction <- function(fullscan_filename, fragmentation_filename, config) {

    ##########################
    ##### Peak Detection #####
    ##########################
    
    create_peak_method <- config$create_peak_method
    if (create_peak_method == 1) {    
        print(paste("Running create_peak_method #1 on", fullscan_filename))
        source('runCreatePeakMethod1.R')    
        peaks <- run_create_peak_method_1(fullscan_filename, config)
    } else if (create_peak_method == 3) {
        print("Running create_peak_method #3")        
        source('runCreatePeakMethod3.R')    
        peaks <- run_create_peak_method_3(fullscan_filename, fragmentation_filename, config)    
    }
    
    ###############################
    ##### Feature Extractions #####
    ###############################
    
    # do further filtering inside create_peaklist() method
    source('createPeakList.R')
    results <- create_peaklist(peaks, config)
    ms1 <- results$ms1
    ms2 <- results$ms2
    
    ########################
    ##### Write Output #####
    ########################
    
    source('writeDataframes.R')
    output_dir <- dirname(fullscan_filename)
    prefix <- basename(file_path_sans_ext(fullscan_filename))
    prefix <- file.path(output_dir, prefix)
    write_output(ms1, ms2, prefix, config)

}