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
  file.remove( dir( "./src", "\\.(o|dll|cu|so|lib|exp)$", full.names = TRUE ) )
}

# TODO ====
# Automatically filled path variables in Makevars files

# Source files containing cuda (device) code and host code with kernel launches
# are found in ./src_cuda, with a .cpp extension so that Rstudio syntax
# highlighting works correctly. These files are copied (if changed) to ./src.

# Windows:
# Cuda sources are compiled into a shared library called cudaR.dll. The
# functions are then imported from this library by the final cuRious.dll. This
# workaround was needed so that mingw64/gcc and NVCC/MSVC compiled objects don't
# need to be linked together.

# Linux:
# The cuda sources are compiled with NVCC into objects, which are then linked
# with regular objects by gcc.

lapply( dir( "./src_cuda", "\\.cpp$" ), function( file ){
  file               <- sub( "\\.cpp$", "", file )
  file.src.path      <- paste0( "./src/", file, ".cu" )
  file.src.cuda.path <- paste0( "./src_cuda/", file, ".cpp" )

  if( file.exists( file.src.path ) ){
    if( md5sum( file.src.cuda.path ) == md5sum( file.src.path ) ){
      return( FALSE )
    }
  }
  file.copy( file.src.cuda.path, file.src.path, overwrite = TRUE )
})

build()
install( args = "--no-lock" )

rm( list = ls( globalenv() ), pos = globalenv() )
gc()
