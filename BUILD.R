# Source this script to build the package

clean.build <- FALSE

# ------------------------------------------------------------------------------

library( devtools )
library( tools )

if( "package:cuRious" %in% search() ){
  detach( "package:cuRious", unload = TRUE )
}

# Pre-clean
if( clean.build ){
  file.remove( dir( "./src",
                    pattern = "(\\.o|\\.dll|\\.cu|\\.so)$",
                    full.names = TRUE ) )
}

# TODO ====
# Automatically filled path variables in Makevars files

# TODO ====
# This might not be required anymore

# Source files containing CUDA gpu code need to be pasted into one single file,
# otherwise the MinGW/g++ linker creates an invalid .dll on windows
cuda.file <- "cudaR.cu"
cuda.src.path <- paste0( "./src/", cuda.file )
cuda.tmp.path <- paste0( "./src_cuda/", cuda.file )

file.remove( dir( "./src_cuda", pattern = "\\.cu$", full.names = TRUE ) )
file.append( cuda.tmp.path, dir( "./src_cuda", full.names = TRUE ) )

if( file.exists( cuda.src.path ) ){
  if( md5sum( cuda.src.path ) != md5sum( cuda.tmp.path ) ){
    file.copy( cuda.tmp.path, cuda.src, overwrite = TRUE )
  }
}else{
  file.copy( cuda.tmp.path, cuda.src.path, overwrite = TRUE )
}

build()
install( args = "--no-lock" )

rm( list = ls( globalenv() ), pos = globalenv() )
gc()
