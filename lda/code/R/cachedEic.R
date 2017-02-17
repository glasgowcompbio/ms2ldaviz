findEIC.cached <- function(msRaw, mz, limit = NULL, rtLimit = NA, headerCache = NULL, floatingRecalibration = NULL,
		peaksCache = NULL)
{
	# calculate mz upper and lower limits for "integration"
	if(all(c("mzMin", "mzMax") %in% names(mz)))
		mzlimits <- c(mz$mzMin, mz$mzMax)
	else
		mzlimits <- c(mz - limit, mz + limit)
	# Find peaklists for all MS1 scans
	if(!is.null(headerCache))
		headerData <- as.data.frame(headerCache)
	else
		headerData <- as.data.frame(header(msRaw))
	# Add row numbering because I'm not sure if seqNum or acquisitionNum correspond to anything really
	if(nrow(headerData) > 0)
		headerData$rowNum <- 1:nrow(headerData)
	else
		headerData$rowNum <- integer(0)
	
	# If RT limit is already given, retrieve only candidates in the first place,
	# since this makes everything much faster.
	if(all(!is.na(rtLimit)))
		headerMS1 <- headerData[
				(headerData$msLevel == 1) & (headerData$retentionTime >= rtLimit[[1]])
						& (headerData$retentionTime <= rtLimit[[2]])
				,]
	else
		headerMS1 <- headerData[headerData$msLevel == 1,]
	if(is.null(peaksCache))
		pks <- mzR::peaks(msRaw, headerMS1$seqNum)
	else
		pks <- peaksCache[headerMS1$rowNum]
		
	# Sum intensities in the given mass window for each scan
	if(is.null(floatingRecalibration))
	{
		headerMS1$mzMin <- mzlimits[[1]]
		headerMS1$mzMax <- mzlimits[[2]]
	}
	else
	{
		headerMS1$mzMin <- mzlimits[[1]] + predict(floatingRecalibration, headerMS1$retentionTime)
		headerMS1$mzMax <- mzlimits[[2]] + predict(floatingRecalibration, headerMS1$retentionTime)
	}
	intensity <- unlist(lapply(1:nrow(headerMS1), function(row)
					{
						peaktable <- pks[[row]]
						sum(peaktable[
										which((peaktable[,1] >= headerMS1[row,"mzMin"]) & (peaktable[,1] <= headerMS1[row,"mzMax"])),
										2])
						
					}))
	return(data.frame(rt = headerMS1$retentionTime, intensity=intensity, scan=headerMS1$acquisitionNum))
}


makePeaksCache <- function(msRaw, headerCache) 
{
	mzR::peaks(msRaw, headerCache$seqNum)
}