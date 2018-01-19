# Dimension encoder
# This function checks for validity too!
# The order of dims is super important!
get.dims <- function( obj ){
  if( is.tensor( obj ) ){
    return( obj$get.dims )
  }

  if( is.vector( obj ) ){
    # R vectors are functionally single column matrices, thats why
    return( c( length( obj ), 1L ) )
  }else if( is.matrix( obj )){
    return( c( nrow( obj ), ncol( obj ) ) )
  }else{
    stop( "Unsupported R object for tensor conversion" )
  }
}

# Location encoder
# Level 0: R object (       host memory, double)
# Level 1: C array  (       host memory, double)
# Level 2: C array  (pinned host memory, float )
# Level 3: C array  (     device memory, float )
get.level <- function( obj ){
  if( is.tensor( obj ) ){
    return( obj$get.level )
  }

  0L
}

# Placeholder object creator
create.dummy <- function( dims ){
  if( dims[2] == 1 ){
    vector( "numeric", length = dims[1] )
  }else{
    matrix( 0, nrow = dims[1], ncol = dims[2] )
  }
}

# Force storage in double
force.double <- function( obj ){
  if( storage.mode( obj ) != "double" ){
    warning( "Supported object is not double precision" )
    storage.mode( obj ) <- "double"
  }

  obj
}

# Force storage in int
force.int <- function( obj ){
  if( storage.mode( obj ) != "integer" ){
    warning( "Supported object is not an integer" )
    storage.mode( obj ) <- "integer"
  }

  obj
}

# Clean global env (and all memory)
clean.global <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}
