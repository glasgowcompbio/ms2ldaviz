create_peaklist <- function(peaks, config) {
    
    ### MS1 ###
    
    # get ms1 peaks
    ms1 <- peaks[which(peaks$msLevel==1),]

    # keep MS1 peaks greater than this intensity
    ms1 <- ms1[which(ms1$intensity >= config$filtering_parameters_MassSpectrometry_related$min_MS1_intensity_wanted),]
    
    # keep MS1 peaks with RT > 3 mins and < 21 mins
    ms1 <- ms1[which(ms1$rt >= config$filtering_parameters_Chromatography_related$rt_start*60),]
    ms1 <- ms1[which(ms1$rt <= config$filtering_parameters_Chromatography_related$rt_end*60),]
    
    ### MS2 ###
    
    # get ms2 peaks
    ms2 <- peaks[which(peaks$msLevel==2),]
    
    # keep ms2 peaks with intensity above noise level
    ms2 <- ms2[which(ms2$intensity>config$filtering_parameters_MassSpectrometry_related$min_MS2_intensity),]
    
    # keep ms2 peaks with parent in filtered ms1 list
    ms2 <- ms2[which(ms2$MSnParentPeakID %in% ms1$peakID),]
    
    # make sure only ms1 peaks with ms2 fragments are kept
    ms1 <- ms1[which(ms1$peakID %in% ms2$MSnParentPeakID),]

    # sort the ms2 dataframe by the intensity column
    ms2 <- ms2[with(ms2, order(-intensity)), ]
    
    # scale the intensities of ms2 peaks to relative intensity
    parent_ids <- ms2$MSnParentPeakID
    for (i in 1:nrow(ms1)) {
        
        print(paste(c("i=", i, "/", nrow(ms1)), collapse=""))
        
        peak_id <- ms1[i, 1]
        matches <- match(as.character(parent_ids), peak_id)
        pos <- which(!is.na(matches))
        # if there's more than one fragment peak
        if (length(pos)>0) {
            # then scale by the relative intensities of the spectrum
            fragment_peaks <- ms2[pos, ]
            fragment_intensities <- fragment_peaks$intensity
            max_intense <- max(fragment_intensities)
            fragment_intensities <- fragment_intensities / max_intense
            ms2[pos, ]$intensity <- fragment_intensities
        }
        
    }
    
    # set the row names to be the same as the peakid
    rownames(ms1) <- ms1$peakID
    rownames(ms2) <- ms2$peakID    
    output <- list("ms1"=ms1, "ms2"=ms2)
    return(output)
    
}
