dive <- function( obj ){
  if( class( obj ) == "externalptr" ){
    return( obj )
  }

  # Vector or scalar
  if( is.vector( obj ) ){
    # Convert to double
    storage.mode( obj ) <- "double"
    ptr <- .Call( "dive_num_vect", obj )

    # Add aditional attributes
    attr( ptr, "type" ) <- "vect"
    attr( ptr, "l" )    <- length( obj )

    return( ptr )
  }

  stop( "Unsupported object type" )
}

surface <- function( ptr ){
  if( class( ptr ) != "externalptr" ){
    return( ptr )
  }

  switch(
    attr( ptr, "type" ),
    vect = .Call( "surface_num_vect", ptr, attr( ptr, "l" ) ),
    stop( "Unsupported object type" )
  )
}
