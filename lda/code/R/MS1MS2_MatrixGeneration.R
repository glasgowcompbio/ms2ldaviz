library(yaml)
source('startFeatureExtraction.R')

process <- function(config_filename) {

    config <- yaml.load_file(config_filename)
    if (config$create_peak_method == 1) {
        
        input_files <- config$input_files$fragmentation_file_m1
        fullscan_filenames <- Sys.glob(input_files)
        print(fullscan_filenames)
        for (fn in fullscan_filenames) {
            start_feature_extraction(fn, NULL, config)
        } 
        
    } else if (config$create_peak_method == 3) {

        fullscan_filenames <- Sys.glob(config$input_files$input_file_forMS1peaks)
        fragmentation_filenames <- Sys.glob(config$input_files$fragmentation_file_mzML)
        for (i in seq_along(fullscan_filenames)) {
            fullscan = fullscan_filenames[i]
            frag = fragmentation_filenames[i]
            start_feature_extraction(fullscan, frag, config)
        }
        
    }

}

### process all urine files ###

process("config/config_urine_pos_1.yml")
process("config/config_urine_neg_1.yml")
process("config/config_urine_pos_3.yml")
process("config/config_urine_neg_3.yml")

### process all beer files ###

process("config/config_beer_pos_1.yml")
process("config/config_beer_neg_1.yml")
process("config/config_beer_pos_3.yml")
process("config/config_beer_neg_3.yml")