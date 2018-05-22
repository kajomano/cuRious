# Clean global env and restart session
clean <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}

.onUnload <- function( libpath ){
  library.dynam.unload( "cuRious", libpath )
}
