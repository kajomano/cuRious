print( paste(
  "sudo",
  "/usr/local/cuda/bin/nvvp",
  paste0( R.home(), "/bin/Rscript" ),
  paste0( getwd(), "/Samples/wip_overlap_bug.R" )
) )

library( cuRious )

invisible( .Call( "cuR_wip_overlap_bug" ) )
