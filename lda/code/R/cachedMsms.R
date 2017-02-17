findMsMsHR.mass.cached <- function(msRaw, mz, limit.coarse, limit.fine, rtLimits = NA, maxCount = NA,
		headerCache = NULL, fillPrecursorScan = FALSE,
		deprofile = getOption("RMassBank")$deprofile, peaksCache = NULL)
{
	eic <- findEIC.cached(msRaw, mz, limit.fine, rtLimits, headerCache=headerCache, peaksCache=peaksCache)
	#	if(!is.na(rtLimits))
	#	{  
	#		eic <- subset(eic, rt >= rtLimits[[1]] & rt <= rtLimits[[2]])
	#	}
	if(!is.null(headerCache))
		headerData <- headerCache
	else
		headerData <- as.data.frame(header(msRaw))
	
	if(fillPrecursorScan == TRUE)
	{
		# reset the precursor scan number. first set to NA, then
		# carry forward the precursor scan number from the last parent scan
		headerData$precursorScanNum <- NA
		headerData[which(headerData$msLevel == 1),"precursorScanNum"] <-
				headerData[which(headerData$msLevel == 1),"acquisitionNum"]
		headerData[,"precursorScanNum"] <- .locf(headerData[,"precursorScanNum"])
		# Clear the actual MS1 precursor scan number again
		headerData[which(headerData$msLevel == 1),"precursorScanNum"] <- 0
	}
	
	# Find MS2 spectra with precursors which are in the allowed 
	# scan filter (coarse limit) range
	findValidPrecursors <- headerData[
			(headerData$precursorMZ > mz - limit.coarse) &
					(headerData$precursorMZ < mz + limit.coarse),]
	# Find the precursors for the found spectra
	validPrecursors <- unique(findValidPrecursors$precursorScanNum)
	# check whether the precursors are real: must be within fine limits!
	# previously even "bad" precursors were taken. e.g. 1-benzylpiperazine
	which_OK <- lapply(validPrecursors, function(pscan)
			{
				pplist <- as.data.frame(
						mzR::peaks(msRaw, which(headerData$acquisitionNum == pscan)))
				colnames(pplist) <- c("mz","int")
				pplist <- pplist[(pplist$mz >= mz -limit.fine)
								& (pplist$mz <= mz + limit.fine),]
				if(nrow(pplist) > 0)
					return(TRUE)
				return(FALSE)
			})
	validPrecursors <- validPrecursors[which(which_OK==TRUE)]
	if(length(validPrecursors) == 0){
		warning("No precursor was detected. It is recommended to try to use the setting fillPrecursorScan: TRUE in the ini-file")
	}
	# Crop the "EIC" to the valid precursor scans
	eic <- eic[eic$scan %in% validPrecursors,]
	# Order by intensity, descending
	eic <- eic[order(eic$intensity, decreasing=TRUE),]
	if(nrow(eic) == 0)
		return(list(list(foundOK = FALSE)))
	if(!is.na(maxCount))
	{
		spectraCount <- min(maxCount, nrow(eic))
		eic <- eic[1:spectraCount,]
	}
	# Construct all spectra groups in decreasing intensity order
	spectra <- lapply(eic$scan, function(masterScan)
			{
				masterHeader <- headerData[headerData$acquisitionNum == masterScan,]
				childHeaders <- headerData[(headerData$precursorScanNum == masterScan) 
								& (headerData$precursorMZ > mz - limit.coarse) 
								& (headerData$precursorMZ < mz + limit.coarse) ,]
				childScans <- childHeaders$acquisitionNum
				
				msPeaks <- mzR::peaks(msRaw, masterHeader$seqNum)
				# if deprofile option is set: run deprofiling
				deprofile.setting <- deprofile
				if(!is.na(deprofile.setting))
					msPeaks <- deprofile.scan(
							msPeaks, method = deprofile.setting, noise = NA, colnames = FALSE
					)
				colnames(msPeaks) <- c("mz","int")
				msmsPeaks <- lapply(childHeaders$seqNum, function(scan)
						{
							pks <- mzR::peaks(msRaw, scan)
							if(!is.na(deprofile.setting))
							{								
								pks <- deprofile.scan(
										pks, method = deprofile.setting, noise = NA, colnames = FALSE
								)
							}
							colnames(pks) <- c("mz","int")
							return(pks)
						}
				)
				return(list(
								foundOK = TRUE,
								parentScan = masterScan,
								parentHeader = masterHeader,
								childScans = childScans,
								childHeaders= childHeaders,
								parentPeak=msPeaks,
								peaks=msmsPeaks
						#xset=xset#,
						#msRaw=msRaw
						))
			})
	names(spectra) <- eic$acquisitionNum
	return(spectra)
}
