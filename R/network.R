# Main network class

nn <- R6Class(
  "nn",
  public = list(
    layers = list()
  )
)

is.nn <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "nn" %in% class( obj )
  })
}

check.nn <- function( ... ){
  if( !all( is.nn( ... ) ) ){
    stop( "Not all objects are nns" )
  }
}
