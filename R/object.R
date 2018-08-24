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
  if( length( dim( obj ) ) && dim( obj ) > .max.array.rank ){
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
    length( obj )
  }else{
    dim( obj )
  }
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
  .Call( "cuR_object_recut", obj, dims )
}

.obj.create <- function( dims, type ){
  obj <- vector( .types[[type]], prod( dims ) )
  .obj.recut( obj, dims )

  obj
}
