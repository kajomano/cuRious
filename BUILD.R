# Source this script to build the package

library( devtools )
if( "package:cuRious" %in% search() ){
  detach( "package:cuRious", unload = TRUE )
}

# Pre-clean
file.remove( dir( "./src", pattern = "(\\.o|\\.dll)$", full.names = TRUE ) )

# Source files containing CUDA gpu code need to be pasted into one single file,
# otherwise the MinGW/g++ linker creates an invalid .dll on windows
cuda.source <- "./src/cuda.cu"

if( file.exists( cuda.source ) ){
  file.remove( cuda.source )
}
file.append( cuda.source, dir( "./src_cuda", full.names = TRUE ) )

build(  )
install( args = "--no-lock" )

require( cuRious )
file.remove( cuda.source )
clean()
