# Package hooks
.cuRious.env <- new.env()

.onLoad <- function( libname, pkgname ){
  assign( "cuda.device.current", .cuda.device.get(), envir = .cuRious.env )
  if( cuda.device.count() == -1 ){
    default <- -1L
  }else{
    default <- 0L
  }
  assign( "cuda.device.default", default, envir = .cuRious.env )
}

.onUnload <- function( libpath ){
  library.dynam.unload( "cuRious", libpath )
}
