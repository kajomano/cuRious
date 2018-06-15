# Source this script to build the package

# Build options
clean.build  <- TRUE
debug.prints <- FALSE
cuda.exclude <- FALSE

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

# Debug ========================================================================
debug.header <- readLines( "./src/common_debug.h", 1 )

if( debug.prints ){
  if( debug.header != "#define DEBUG_PRINTS 1" ){
    writeLines( "#define DEBUG_PRINTS 1", "./src/common_debug.h" )
  }
}else{
  if( debug.header != "// #define DEBUG_PRINTS 1" ){
    writeLines( "// #define DEBUG_PRINTS 1", "./src/common_debug.h" )
  }
}

# Cuda =========================================================================
cuda.header <- readLines( "./src/common_cuda.h", 1 )

if( cuda.exclude ){
  if( cuda.header != "#define CUDA_EXCLUDE 1" ){
    writeLines( "#define CUDA_EXCLUDE 1", "./src/common_cuda.h" )
  }

  # TODO ====
  # Didnt remember the windows extensions for cuda dll files, extend regexp
  file.remove( dir( "./src", "\\.(cu|lib|exp)$", full.names = TRUE ) )
}else{
  if( cuda.header != "// #define CUDA_EXCLUDE 1" ){
    writeLines( "// #define CUDA_EXCLUDE 1", "./src/common_cuda.h" )
  }

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
}

# Makevars =====================================================================
file.remove( dir( "./src/Makevars", full.names = TRUE ) )
card.list <- list()
makevars.path <- "./Makevars/Makevars"

if( Sys.info()["sysname"] == "Windows" ){
  makevars.path <- paste0( makevars.path, "_windows" )

  if( cuda.exclude ){
    makevars.path <- paste0( makevars.path, "_nocuda" )
  }else{
    makevars.path <- paste0( makevars.path, "_cuda" )

    card.list$CUDA_HOME <- Sys.getenv()[[ "CUDA_PATH" ]]
    card.list$CUDA_HOME <- gsub( "\\", "/", card.list$CUDA_HOME, fixed = TRUE )

    card.list$VC_HOME   <- Sys.getenv()[[ "VS140COMNTOOLS" ]]
    card.list$VC_HOME   <- gsub( "\\", "/", card.list$VC_HOME, fixed = TRUE )
    card.list$VC_HOME   <- gsub( "Common7/Tools/", "VC", card.list$VC_HOME,
                                 fixed = TRUE )

    card.list$CUR_SRC   <- paste0( getwd(), "/src" )
  }
}else{
  if( cuda.exclude ){
    makevars.path <- paste0( makevars.path, "_nocuda" )
  }else{
    makevars.path <- paste0( makevars.path, "_cuda" )

    stop( "Check this" )
    card.list$CUDA_HOME <- Sys.getenv()[[ "CUDA_PATH" ]]
  }
}

makevars <- readLines( makevars.path )
makevars <- sapply( makevars, function( line ){
  if( length( card.list ) ){
    for( i in 1:length( card.list ) ){
      line <- sub(
        paste0( "%", names( card.list )[[i]], "%" ),
        paste0( "\"", card.list[[i]], "\"" ),
        line,
        fixed = TRUE
      )
    }
  }

  line
})
writeLines( makevars, "./src/Makevars" )

# Build ========================================================================
build()
install( args = c( "--no-lock", "--no-multiarch" ) )

# Clean ========================================================================
rm( list = ls( globalenv() ), pos = globalenv() )
gc()
