# Dimension encoder
# This function checks for validity too!
# The order of dims is super important!
get.dims <- function( obj ){
  switch(
    class( obj )[[1]],
    tensor     = obj$get.dims,
    tensor.ptr = c( attr( obj, "dim0" ), attr( obj, "dim1" ) ),
    matrix     = c( nrow( obj ), ncol( obj ) ),
    numeric    = c( length( obj ), 1L ),
    integer    = c( length( obj ), 1L ),
    stop( "Unknown object" )
  )
}

# Location encoder
# Level 0: R object (        host memory, double )
# Level 1: C array  (        host memory, float  )
# Level 2: C array  ( pinned host memory, float  )
# Level 3: C array  (      device memory, float  )
get.level <- function( obj ){
  switch(
    class( obj )[[1]],
    tensor     = obj$get.level,
    tensor.ptr = attr( obj, "level" ),
    matrix     = 0L,
    numeric    = 0L,
    integer    = 0L,
    stop( "Unknown object" )
  )
}

# Placeholder object creator
create.dummy <- function( dims, level = 0 ){
  if( prod( dims ) > 2^32-1 ){
    # TODO ====
    # Use long int or the correct R type to remove this constraint
    stop( "Object is too large" )
  }

  res <- switch(
    as.character(level),
    `0` = {
      if( dims[2] == 1 ){
        vector( "numeric", length = dims[1] )
      }else{
        matrix( 0, nrow = dims[1], ncol = dims[2] )
      }
    },
    `1` = .Call( "cuR_create_tensor_1", dims ),
    `2` = .Call( "cuR_create_tensor_2", dims ),
    `3` = .Call( "cuR_create_tensor_3", dims ),
    stop( "Invalid level" )
  )
}

# Force storage in double
force.double <- function( obj ){
  if( is.vector( obj ) || is.matrix( obj ) ){
    if( storage.mode( obj ) != "double" ){
      warning( "Supported object is not double precision" )
      storage.mode( obj ) <- "double"
    }
  }

  obj
}

# Force storage in int
force.int <- function( obj ){
  if( is.vector( obj ) || is.matrix( obj ) ){
    if( storage.mode( obj ) != "integer" ){
      warning( "Supported object is not an integer" )
      storage.mode( obj ) <- "integer"
    }
  }

  obj
}

# Clean global env (and all memory)
clean.global <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}
