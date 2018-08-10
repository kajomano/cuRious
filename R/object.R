# .Calls: src/object.cpp
# Functions on native R objects ====
is.obj <- function( obj ){
  # Mode check
  if( switch(
    storage.mode( obj ),
    double  = FALSE,
    integer = FALSE,
    logical = FALSE,
    TRUE
  ) ){
    return( FALSE )
  }

  # Dim check
  if( length( dim( obj ) ) && dim( obj ) > 2 ){
    return( FALSE )
  }

  TRUE
}

check.obj <- function( obj ){
  if( !is.obj( obj ) ){
    stop( "Not an object" )
  }
}

obj.dims <- function( obj ){
  check.obj( obj )

  if( is.null( dim( obj ) ) ){
    dims <- length( obj )
  }else{
    dims <- dim( obj )
  }

  c( dims, rep( 1L, 2 - length( dims ) ) )
}

obj.type <- function( obj ){
  check.obj( obj )

  switch(
    storage.mode( obj ),
    double  = "n",
    integer = "i",
    logical = "l"
  )
}

.obj.duplicate <- function( obj ){
  .Call( "cuR_object_duplicate", obj )
}

.obj.recut <- function( obj, dims ){
  if( dims[[2]] == 1L ){
    dims <- dims[[1]]
  }else{
    dims <- dims
  }

  .Call( "cuR_object_recut", obj, dims )
}

.obj.create <- function( dims, type ){
  obj <- vector( types[[type]], prod( dims ) )
  .obj.recut( obj, dims )

  obj
}

# TODO ====
# Remove this after row subsets are implemented

# Subset an object with the same rules as cuRious does
# ( Vectors normally, matrices by column )
obj.subset <- function( obj, subset = NULL ){
  obj    <- check.obj( obj )

  if( is.null( subset ) ){
    return( obj )
  }

  subset <- check.obj( subset )

  if( obj.type( subset ) != "i" || obj.dims( subset )[[1]] != 1L ){
    stop( "Invalid subset" )
  }

  if( obj.dims( obj )[[1]] == 1L ){
    obj[ subset ]
  }else{
    obj[, subset ]
  }
}
