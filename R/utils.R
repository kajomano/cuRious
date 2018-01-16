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

# Placeholder object creator
create.dummy <- function( dims ){
  if( dims[2] == 1 ){
    vector( "numeric", length = dims[1] )
  }else{
    matrix( 0, nrow = dims[1], ncol = dims[2] )
  }
}

# Clean global env (and all memory)
clean.global <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}
