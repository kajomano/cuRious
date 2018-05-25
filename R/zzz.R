# Package hooks
.cuRious.env <- new.env()

.onLoad <- function( libname, pkgname ){
  assign( "cuda.device.current", .cuda.device.get(), envir = .cuRious.env )
  assign( "cuda.device.default", 0L, envir = .cuRious.env )
}

.onUnload <- function( libpath ){
  library.dynam.unload( "cuRious", libpath )
}