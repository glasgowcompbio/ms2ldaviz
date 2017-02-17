############
##Define xcmsSetFragments function for correct peak picking of must abundant MS2 spectrum associated with MS1 peaks picked by xcmsSet.
############

## fixed version of xcmsFragments that correctly assigns ms2 spectra to ms1 peaks and preserves any xcmsSet group info
## inputs:
## xs = an xcmsSet object.  
## cdf.corrected = logical TRUE or FALSE.  Default = FALSE. Was a m/z corrected .cdf file used for xs peak picking?
## min.rel.int = minimum relative intensity of ms2 peaks (default = 0.01)
## max.frags = maximum number of ms2 fragments associated within a ms1 parent (default = 8)
## msnSelect = alternate criteria for selecting which ms2 spectra belong to which ms1 precursor:
##	precursor_int = most intense precursor ion (default; typical for Thermo data dependent acquisition)
##	ms1_rt = closest precursor rt
##	all = any ms2 peaks linked to ms1 precursor, with any duplicated ms2 peaks within match.ppm (default = 300) grouped together
## specFilter = filter criteria for removing ms2 noise peaks
##	none = no filtering (default)
##	specPeaks = xcms specPeaks function using sn = 3 and mz.gap = 0.2 as defaults
##	cor = correlation based filtering with minimum r = 0.75
## mindiff = minimum difference between precursor and any ms2 fragment (default = 10)

xcmsSetFragments <- function(xs, cdf.corrected = FALSE, min.rel.int = 0.01, max.frags = 5000, msnSelect = c("precursor_int"), 
    specFilter = c("specPeaks"), match.ppm = 7, sn = 3, mzgap = 0.005, min.r = 0.75, min.diff = 10) {
    
    require("xcms")
    require("Hmisc")
    
    msnSelect <- match.arg(msnSelect, c("precursor_int", "ms1_rt", "all"))
    specFilter <- match.arg(specFilter, c("none", "specPeaks", "cor"))
    
    if (msnSelect != "all" & specFilter == "cor") {
        stop("correlation filtering is not possible with single spectrum selection!", "\n")
    }
    
    if (class(xs) == "xcmsSet") {
        ms1peaks <- peaks(xs)
    } else {
        stop("input is not an xcmsSet")
    }
    
    # create new xcmsFragments object
    object <- new("xcmsFragments")
    
    # msnSpecs without ms1-parentspecs
    numAloneSpecs <- 0
    
    numMs1Peaks <- length(ms1peaks[, "mz"])
    npPeakID <- 1:numMs1Peaks
    npMSnParentPeakID <- rep(0, numMs1Peaks)
    npMsLevel <- rep(1, numMs1Peaks)
    
    npMz <- ms1peaks[, "mz"]
    npMinMz <- ms1peaks[, "mzmin"]
    npMaxMz <- ms1peaks[, "mzmax"]
    
    npRt <- ms1peaks[, "rt"]
    npMinRt <- ms1peaks[, "rtmin"]
    npMaxRt <- ms1peaks[, "rtmax"]
    
    npIntensity <- ms1peaks[, "maxo"]
    npSample <- ms1peaks[, "sample"]
    npCollisionEnergy <- rep(0, numMs1Peaks)
        
    # PeakNr+1 is the beginning peakindex for msn-spectra
    PeakNr <- numMs1Peaks
    
    # extract xcmsRaw files with msn spectra
    paths <- length(xs@filepaths)
    for (NumXcmsPath in 1:paths) {
        cat("Processing file ", basename(xs@filepaths[NumXcmsPath]), " (", NumXcmsPath, " of ", paths, ")", "\n", 
            sep = "")
        xcmsRawPath <- xs@filepaths[NumXcmsPath]
        xr <- xcmsRaw(xcmsRawPath, includeMSn = TRUE)
        
        # If an xcmsSet with corrected RTs, adjust xr scantime and use linear interpolation to adjust msnRt
        rawRT <- xs@rt$raw[[NumXcmsPath]]
        corrRT <- xs@rt$corrected[[NumXcmsPath]]
        if (!all(rawRT == corrRT)) {
            xr@scantime <- corrRT
            xr@msnRt <- approx(x = rawRT, y = corrRT, xout = xr@msnRt, rule = 2)$y
        }
        
        # If a mass-corrected .cdf file was used, apply same correction to xr@env$mz and xr@msnPrecursorMz
        if (cdf.corrected) {
            xr.cdf <- xcmsRaw(gsub(".mzXML", ".cdf", xcmsRawPath, fixed = T), includeMSn = F)
            mz.offset <- mean(xr@env$mz - xr.cdf@env$mz)
            xr@env$mz <- xr@env$mz - mz.offset
            
            precursor.prec <- max(sapply(xr@msnPrecursorMz, function(x) {
                nchar(strsplit(as.character(x), "\\.")[[1]][2])
            }))
            xr@msnPrecursorMz <- round(xr@msnPrecursorMz - mz.offset, precursor.prec)
        }
        
        
        # identify msn scans for every precursor
        precursor.mz <- xr@msnPrecursorMz
        msn.rt <- xr@msnRt
        
        # extract a composite msn spectrum where it exists for every ms1 peak
        for (i in 1:nrow(ms1peaks)) {
            ActualParentPeakID <- 0
            if (ms1peaks[i, "sample"] == NumXcmsPath) {
                # indices of all msn scans where msn precursor mass is within the ms1 peak mz and rt range
                msn.idx <- which(precursor.mz >= npMinMz[i] & precursor.mz <= npMaxMz[i] & msn.rt >= npMinRt[i] & 
                  msn.rt <= npMaxRt[i])
                
                if (length(msn.idx) > 0) {
                  MzTable <- NULL
                  ActualParentPeakID <- i
                  
                  # Single msn spectrum with highest precursor mass intensity
                  if (msnSelect == "precursor_int") {
                    precursor.int <- xr@msnPrecursorIntensity[msn.idx]
                    msn.id <- msn.idx[which.max(precursor.int)]
                    representative.msn.id <- msn.id
                    if (msn.id < length(xr@msnScanindex)) {
                      start.id <- xr@msnScanindex[msn.id] + 1
                      end.id <- xr@msnScanindex[msn.id + 1]
                    } else {
                      start.id <- xr@msnScanindex[msn.id] + 1
                      end.id <- xr@env$msnMz
                    }
                    MzTable <- new("matrix", ncol = 2, nrow = length(xr@env$msnMz[start.id:end.id]), data = c(xr@env$msnMz[start.id:end.id], 
                      xr@env$msnIntensity[start.id:end.id]))
                    colnames(MzTable) <- c("mz", "intensity")
                  }
                  
                  # Single msn spectrum with closest rt match to ms1peak rt
                  if (msnSelect == "ms1_rt") {
                    msn.id <- msn.idx[which.min(abs(npRt[i] - msn.rt[msn.idx]))]
                    representative.msn.id <- msn.id
                    if (msn.id < length(xr@msnScanindex)) {
                      start.id <- xr@msnScanindex[msn.id] + 1
                      end.id <- xr@msnScanindex[msn.id + 1]
                    } else {
                      start.id <- xr@msnScanindex[msn.id] + 1
                      end.id <- xr@env$msnMz
                    }
                    MzTable <- new("matrix", ncol = 2, nrow = length(xr@env$msnMz[start.id:end.id]), data = c(xr@env$msnMz[start.id:end.id], 
                      xr@env$msnIntensity[start.id:end.id]))
                    colnames(MzTable) <- c("mz", "intensity")
                    MzTable <- MzTable[which(MzTable[, "intensity"]/max(MzTable[, "intensity"]) > min.rel.int), , 
                      drop = F]
                  }
                  
                  # if all msn spectra associated with the ms1 peak are to be used
                  if (msnSelect == "all") {
                    # representative msn.id = closest RT
                    representative.msn.id <- msn.idx[which.min(abs(npRt[i] - msn.rt[msn.idx]))]
                    
                    c.MzTable <- NULL
                    count <- 0
                    # extract msn mz and intensity values for every in-range msn scan
                    for (msn.id in msn.idx) {
                      count <- count + 1
                      if (msn.id < length(xr@msnScanindex)) {
                        start.id <- xr@msnScanindex[msn.id] + 1
                        end.id <- xr@msnScanindex[msn.id + 1]
                      } else {
                        start.id <- xr@msnScanindex[msn.id] + 1
                        end.id <- xr@env$msnMz
                      }
                      
                      c.MzTable <- rbind(c.MzTable, cbind(msn.id = rep(msn.id, length(xr@env$msnMz[start.id:end.id])), 
                        mz = xr@env$msnMz[start.id:end.id], intensity = xr@env$msnIntensity[start.id:end.id]))
                    }
                    
                    # create an intensity-weighted mz vector and an intensity matrix filled within match.ppm, with missing mz vals as
                    # NA
                    weighted.mz <- numeric()
                    msn.intensity <- list()
                    int <- c.MzTable[, "intensity"]
                    count <- 0
                    while (any(!is.na(int))) {
                      count <- count + 1
                      max.idx <- which.max(int)
                      mzmin <- c.MzTable[max.idx, "mz"] - (match.ppm/1e+06 * c.MzTable[max.idx, "mz"])
                      mzmax <- c.MzTable[max.idx, "mz"] + (match.ppm/1e+06 * c.MzTable[max.idx, "mz"])
                      mz.idx <- which(c.MzTable[, "mz"] >= mzmin & c.MzTable[, "mz"] <= mzmax)
                      
                      weighted.mz[count] <- weighted.mean(c.MzTable[mz.idx, "mz"], c.MzTable[mz.idx, "intensity"])
                      msn.intensity[[count]] <- c.MzTable[mz.idx, , drop = F][match(msn.idx, c.MzTable[mz.idx, "msn.id", 
                        drop = F]), "intensity"]
                      int[mz.idx] <- NA
                    }
                    msn.intensity <- do.call("cbind", msn.intensity)
                    
                    # calculate mean intensities, penalizing NA intensities by assigning them as 0
                    msn.intensity.zeroed <- msn.intensity
                    msn.intensity.zeroed[which(is.na(msn.intensity.zeroed))] <- 0
                    mean.msn.intensity <- apply(msn.intensity.zeroed, 2, mean)
                    MzTable <- cbind(mz = weighted.mz, intensity = mean.msn.intensity)
                  }
                  
                  if (specFilter == "none") {
                    npeaks <- MzTable[order(MzTable[, "intensity"], decreasing = T), , drop = F]
                  }
                  if (specFilter == "specPeaks") {
                    MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
                    npeaks <- specPeaks(MzTable, sn = sn, mzgap = mzgap)[, c("mz", "intensity"), drop = F]
                  }
                  if (specFilter == "cor") {
                    if (nrow(msn.intensity) > 4) {
                      precursor.int <- xr@msnPrecursorIntensity[msn.idx]
                      cor.mat <- cbind(precursor.int, msn.intensity)
                      cor.result <- rcorr(cor.mat)
                      cor.summary <- cbind(mz = weighted.mz, intensity = mean.msn.intensity, r = cor.result$r[2:nrow(cor.result$r), 
                        1], n = cor.result$n[2:nrow(cor.result$n), 1], p = cor.result$P[2:nrow(cor.result$P), 1])
                      npeaks <- cor.summary[which(cor.summary[, "r"] > min.r & cor.summary[, "n"] >= 4 & cor.summary[, 
                        "p"] < 0.05), c("mz", "intensity"), drop = F]
                    } else {
                      cat("ms1 peaktable row ", i, " has only ", nrow(msn.intensity), " msn spectra; reverting to specPeaks", 
                        "\n", sep = " ")
                      MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
                      npeaks <- specPeaks(MzTable, sn = sn, mzgap = mzgap)[, c("mz", "intensity"), drop = F]
                    }
                  }
                  
                  if (nrow(npeaks) > 0) {
                    # only keep peaks with relative intensities >= min.rel.int
                    npeaks <- npeaks[which(npeaks[, "intensity"]/max(npeaks[, "intensity"]) >= min.rel.int), , drop = F]
                    # only keep peaks whose masses are smaller than precursor mass - min.diff
                    npeaks <- npeaks[which(npeaks[, "mz"] < (npMz[i] - min.diff)), , drop = F]
                    if (nrow(npeaks) > 0) {
                      # only keep a maximum of max.frags peaks
                      npeaks <- npeaks[1:min(max.frags, nrow(npeaks)), , drop = F]
                      for (numPeaks in 1:nrow(npeaks)) {
                        # for every picked msn peak
                        PeakNr <- PeakNr + 1
                        # increasing peakid
                        npPeakID[PeakNr] <- PeakNr
                        npMSnParentPeakID[PeakNr] <- ActualParentPeakID
                        npMsLevel[PeakNr] <- xr@msnLevel[representative.msn.id]
                        npRt[PeakNr] <- xr@msnRt[representative.msn.id]
                        npMz[PeakNr] <- npeaks[numPeaks, "mz"]
                        npIntensity[PeakNr] <- npeaks[numPeaks, "intensity"]
                        npSample[PeakNr] <- NumXcmsPath
                        npCollisionEnergy[PeakNr] <- xr@msnCollisionEnergy[representative.msn.id]
                      }
                    }
                  }
                }
            }
        }
        if (ActualParentPeakID == 0) {
            numAloneSpecs <- numAloneSpecs + 1
        }
    }
    
    fragmentColnames <- c("peakID", "MSnParentPeakID", "msLevel", "rt", "mz", "intensity", "Sample", "GroupPeakMSn", 
        "CollisionEnergy")
    npGroupPeakMSn <- rep(0, length(npSample))
    # add group information if it exists
    gv <- groupval(xs)
    if (length(gv) > 0) {
        for (i in 1:nrow(gv)) {
            npGroupPeakMSn[npPeakID %in% gv[i, ]] <- i
            npGroupPeakMSn[npMSnParentPeakID %in% gv[i, ]] <- i
        }
    }
    
    object@peaks <- new("matrix", nrow = length(npMz), ncol = length(fragmentColnames), data = c(npPeakID, npMSnParentPeakID, 
        npMsLevel, npRt, npMz, npIntensity, npSample, npGroupPeakMSn, npCollisionEnergy))
    colnames(object@peaks) <- fragmentColnames
    
    cat(length(npPeakID), "Peaks picked,", numAloneSpecs, "MSn-Specs ignored.\n")
    object
}
