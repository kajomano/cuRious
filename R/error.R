# TODO: get calling funcion name for a better error print

error.undef.op <- function( ... ){
  stop( "Undefined operation" )
}

error.inv.type <- function( ... ){
  stop( "Invalid type" )
}

error.unmatch.dim <- function( ... ){
  stop( "Not matching dimensions" )
}
