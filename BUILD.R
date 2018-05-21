# Source this script to build the package

clean.build <- TRUE

# ------------------------------------------------------------------------------

library( devtools )
library( tools )

if( "package:cuRious" %in% search() ){
  detach( "package:cuRious", unload = TRUE )
}

# Pre-clean
if( clean.build ){
  file.remove( dir( "./src",
                    pattern = "(\\.o|\\.dll|\\.cu)$",
                    full.names = TRUE ) )
}

# Source files containing CUDA gpu code need to be pasted into one single file,
# otherwise the MinGW/g++ linker creates an invalid .dll on windows
cuda.src <- "./src/cuda.cu"
cuda.tmp <- "./src_cuda/cuda.cu"

if( file.exists( cuda.tmp ) ){
  file.remove( cuda.tmp )
}
file.append( cuda.tmp, dir( "./src_cuda", full.names = TRUE ) )

if( file.exists( cuda.src ) ){
  if( md5sum( cuda.src ) != md5sum( cuda.tmp ) ){
    file.copy( cuda.tmp, cuda.src, overwrite = TRUE )
  }
}else{
  file.copy( cuda.tmp, cuda.src, overwrite = TRUE )
}

build()
install( args = "--no-lock" )

rm( list = ls( globalenv() ), pos = globalenv() )
gc()
