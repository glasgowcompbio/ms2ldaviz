write_output <- function(ms1, ms2, prefix, config) {
    
    # construct the output filenames
    ms1_out <- paste(c(prefix, '_ms1.csv'), collapse="")
    ms2_out <- paste(c(prefix, '_ms2.csv'), collapse="")
    
    # write stuff out
    write.table(ms1, file=ms1_out, col.names=NA, row.names=T, sep=",")
    write.table(ms2, file=ms2_out, col.names=NA, row.names=T, sep=",")    

}