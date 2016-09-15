source('cachedEic.R')
source('cachedMsms.R')

### This is the peak detection workflow based on the RMassBank's script from Emma ###
run_create_peak_method_3 <- function(MS1file, fragmentation_file, config) {

    # various tolerance parameters to configure
    dppm <- config$ms1_ms2_pairing_parameters$dppm
    rt_window <- c(config$ms1_ms2_pairing_parameters$rt_window_from, config$ms1_ms2_pairing_parameters$rt_window_to)
    ms_msms_cut <- config$filtering_parameters_MassSpectrometry_related$ms_msms_cut
    select_most_intense <- config$ms1_ms2_pairing_parameters$select_most_intense
    rt_ms1_ms2_difference <- config$ms1_ms2_pairing_parameters$rt_ms1_ms2_difference
    rt_start_minutes <- config$filtering_parameters_Chromatography_related$rt_start_before_pairing
    rt_end_minutes <- config$filtering_parameters_Chromatography_related$rt_end_before_pairing
    rt_start_peak_minutes <- config$filtering_parameters_Chromatography_related$rt_start_peak_before_pairing
    minimum_MS1_int_for_pairing <- config$filtering_parameters_MassSpectrometry_related$min_MS1_intensity
    
    # If we specify the mzXML file for the full scan data, then we want to do peak picking etc.
    if (grepl("mzXML", MS1file)) {
        
        # input file for MS1 peaks is an mzXML file
        xset_full <- xcmsSet(files=MS1file, method=config$MS1_XCMS_peakpicking_settings$method, ppm=config$MS1_XCMS_peakpicking_settings$ppm, snthresh=config$MS1_XCMS_peakpicking_settings$snthres, peakwidth=c(config$MS1_XCMS_peakpicking_settings$peakwidth_from,config$MS1_XCMS_peakpicking_settings$peakwidth_to),
                             prefilter=c(config$MS1_XCMS_peakpicking_settings$prefilter_from,config$MS1_XCMS_peakpicking_settings$prefilter_to), mzdiff=config$MS1_XCMS_peakpicking_settings$mzdiff, integrate=config$MS1_XCMS_peakpicking_settings$integrate, fitgauss=config$MS1_XCMS_peakpicking_settings$fitgauss, verbose.column=config$MS1_XCMS_peakpicking_settings$verbose.column)
        xset_full <- group(xset_full)
        
        # apply some initial filtering
        peak_info <- peaks(xset_full)
        peak_info <- as.data.frame(peak_info)
        peak_info <- peak_info[which(peak_info$rt >= rt_start_minutes*60),]
        peak_info <- peak_info[which(peak_info$rt <= rt_end_minutes*60),]
        peak_info <- peak_info[which(peak_info$rtmin >= rt_start_peak_minutes*60),]
        peak_info <- peak_info[which(peak_info$maxo >= minimum_MS1_int_for_pairing),]
        
        # take only the columns we need
        keeps <- c("mz", "rt", "maxo", "rtmin")
        peak_info <- peak_info[, keeps] 
        colnames(peak_info)[3] <- "int" # rename maxo to int
        
    } else { # otherwise just load the CSV file directly, assuming that it has been pre-filtered
        
        peak_info <- read.csv(MS1file)
        peak_info <- as.data.frame(peak_info)
        
    }
    
    # sort by mz
    peak_info <- peak_info[with(peak_info, order(mz)), ]
    row.names(peak_info) <- NULL  
    
    # open fragmentation file
    RmbDefaultSettings()
    f <- openMSfile(fragmentation_file) 
    h <- header(f)
    
    num_ms1_peaks <- nrow(peak_info)
    np_rt <- peak_info$rt
    np_mz <- peak_info$mz
    np_intensity <- peak_info$int
    
    # get the MS/MS
    print("Finding MS2 peaks")
    mzrt <- cbind(np_mz, np_rt)
    c <- makePeaksCache(f,h)
    if (select_most_intense) {
        msms <- apply(mzrt, 1, function(row) {
            spec <- findMsMsHR.mass.cached(f, row[[1]], 0.5, ppm(row[[1]], dppm, p=TRUE),
                                           rtLimits=row[[2]]+rt_window, maxCount=1, 
                                           headerCache=h, peaksCache=c)
            spec <- spec[[1]]
            mzLimits <- list(
                mzMin=row[1]-ppm(row[1], 10, p=TRUE),
                mzCenter=row[1],
                mzMax=row[1]+ppm(row[1], 10, p=TRUE))
            spec$mz <- mzLimits
            print(paste(c("Finding MS2 peaks for mz=", row[1], " rt=", row[2]), collapse=""))
            return(spec)
        })
        
    } else {
        
        msms <- apply(mzrt, 1, function(row) {
            
            rtApex <- row[[2]]
            spec <- findMsMsHR.mass.cached(f, row[[1]], 0.5, ppm(row[[1]], dppm, p=TRUE),
                                           rtLimits=rtApex+rt_window, headerCache=h, peaksCache=c)
            # find the one closest to rt apex
            if(spec[[1]]$foundOK) {
                rtMsms <- unlist(lapply(spec, function(s) s$childHeaders$retentionTime))
                best <- which.min(abs(rtMsms-rtApex))
                spec<-spec[[best]]
            } else {
                spec <- spec[[1]]
            }
            
            mzLimits <- list(
                mzMin=row[1]-ppm(row[1], dppm, p=TRUE),
                mzCenter=row[1],
                mzMax=row[1]+ppm(row[1], dppm, p=TRUE))
            spec$mz <- mzLimits
            
            print(paste(c("Finding MS2 peaks for mz=", row[1], " rt=", row[2]), collapse=""))
            return(spec)
            
        })
        
    }
    stopifnot(num_ms1_peaks == length(msms))
    
    # make an empty dataframe
    peaks_colnames <- c("peakID", "MSnParentPeakID", "msLevel", "rt", "mz", "intensity", 
                        "Sample", "GroupPeakMSn", "CollisionEnergy")
    peaks <- data.frame(t(rep(NA, length(peaks_colnames))))
    colnames(peaks) <- peaks_colnames      
    
    # loop over all the MS1 peaks and append to the dataframe
    peak_id <- 0
    sample_idx <- 1
    group_peak_msn <- 0
    collision_energy <- 0
    total_ms1_accepted <- 0
    for(i in 1:num_ms1_peaks) { 
        
        print(paste(c("i=", i, "/", num_ms1_peaks), collapse=""))
        
        t <- msms[[i]]
        if (t$foundOK == FALSE) {
            next
        }
        
        # HCD:
        cut <- ms_msms_cut
        cut_ratio <- 0
        shot <- as.data.frame(t$peaks[[1]])
        shot <- shot[(shot$int >= cut) & (shot$int > max(shot$int) * 
                                              cut_ratio), ]
        # remove satellite peaks
        shot <- filterPeakSatellites(shot)
        
        # skip those without any MS2
        num_ms2 <- nrow(shot)
        if (num_ms2 == 0) {
            next
        }
        
        # check that the difference between RT of full scan in 
        # fragmentation file and RT of MS1 feature from peak list
        # isn't too big
        rt_ms <- np_rt[i]
        rt_ms2_from_fragmentation <- t$childHeaders$retentionTime[[1]] # use the real RT from the MSMS
        diff <- abs(rt_ms-rt_ms2_from_fragmentation)
        if (diff > rt_ms1_ms2_difference) {
            next # if too big, then skip
        }
        
        # append MS1 info
        peak_id <- peak_id + 1
        ms1_parent_peak_id <- 0
        ms1_level <- 1
        ms1_rt <- np_rt[i]
        ms1_mz <- np_mz[i]
        ms1_intensity <- np_intensity[i]
        new_row <- c(peak_id, ms1_parent_peak_id, ms1_level, ms1_rt, ms1_mz, ms1_intensity, 
                     sample_idx, group_peak_msn, collision_energy)
        peaks <- rbind(peaks, new_row)  
        
        # set the ms2 info
        ms2_parent_peakids <- rep(peak_id, num_ms2)    
        ms2_peakid_start <- peak_id+1
        ms2_peakid_end <- ms2_peakid_start + num_ms2-1
        ms2_peakids <- ms2_peakid_start:ms2_peakid_end
        peak_id <- tail(ms2_peakids, n=1) # set last peakid used, for the next iteration
        
        ms2_masses <- shot$mz
        ms2_intensities <- shot$int
        ms2_rts <- rep(rt_ms, num_ms2)
        
        # other stuff that are always fixed
        ms2_levels <- rep(2, num_ms2)
        ms2_samples <- rep(1, num_ms2)
        ms2_group <- rep(0, num_ms2)
        ms2_energy <- rep(0, num_ms2)    
        
        # append MS2 info
        ms2_df <- data.frame(ms2_peakids, ms2_parent_peakids, ms2_levels, 
                             ms2_rts, ms2_masses, ms2_intensities, 
                             ms2_samples, ms2_group, ms2_energy)
        colnames(ms2_df) <- peaks_colnames
        peaks <- rbind(peaks, ms2_df)
         
        total_ms1_accepted <- total_ms1_accepted+1
        
    }
 
    print(paste(c("total_ms1_accepted=", total_ms1_accepted, "/", num_ms1_peaks), collapse=""))
    peaks <- peaks[-1, ] # delete first row of all NAs
    return(peaks)
    
}