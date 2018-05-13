# Clean global env and restart session
clean <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
}

# .onUnload <- function( libpath ){
#   library.dynam.unload( "cuRious", "C:/Users/Kajomano/Documents/R/win-library/3.4/cuRious/libs/x64/cuRious.dll" )
# }
