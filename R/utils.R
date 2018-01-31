# Tensor types
tensor.types <- c( f = "numeric", i = "integer", b = "logical" )

# Placeholder object creator
create.obj <- function( dims, level = 0, type = "numeric" ){
  type <- match.arg( type, tensor.types )

  if( prod( dims ) > 2^32-1 ){
    # TODO ====
    # Use long int or the correct R type to remove this constraint
    stop( "Object is too large" )
  }

  switch(
    as.character(level),
    `0` = {
      if( dims[2] == 1 ){
        vector( type, dims[1] )
      }else{
        matrix( vector( type, 1 ), nrow = dims[1], ncol = dims[2] )
      }
    },
    `1` = .Call( paste0("cuR_create_tensor_1", names(type)), dims ),
    `2` = .Call( paste0("cuR_create_tensor_2", names(type)), dims ),
    `3` = .Call( paste0("cuR_create_tensor_3", names(type)), dims ),
    stop( "Invalid level" )
  )
}

# Duplicate (hard copy) utility function, just like in data.table
duplicate.obj <- function( obj ){
  cl <- class( obj )[[1]]
  if( cl == "tensor" ){
    tensor$new( obj )
  }else if( cl %in% c( "tensor.ptr", "matrix", "numeric", "integer", "logical" ) ){
    duplicate <- create.obj( get.dims( obj ), get.level( obj ), get.type( obj ) )
    transfer( obj, duplicate )
    duplicate
  }else{
    stop( "Invalid object" )
  }
}

# Remove an object, completely freeing up allocated memory space
destroy.obj <- function( obj ){
  obj.ptr <- switch(
    class( obj )[[1]],
    tensor     = obj$get.tensor,
    tensor.ptr = obj,
    matrix     = obj,
    numeric    = obj,
    integer    = obj,
    logical    = obj,
    stop( "Invalid object" )
  )

  switch(
    as.character( get.level( obj.ptr ) ),
    `1` = .Call( "cuR_destroy_tensor_1", obj.ptr ),
    `2` = .Call( "cuR_destroy_tensor_2", obj.ptr ),
    `3` = .Call( "cuR_destroy_tensor_3", obj.ptr )
  )

  # Assign NULL
  assign( as.character(substitute(obj) ),
          NULL,
          envir = parent.frame(),
          inherits = TRUE )
}

# Get storage type
get.type <- function( obj ){
  switch(
    class( obj )[[1]],
    tensor     = obj$get.type,
    tensor.ptr = attr( obj, "type" ),
    matrix     = {
      mode <- storage.mode( matrix )
      if( !(mode %in% tensor.types) ){
        stop("Invalid type")
      }
      mode
    },
    numeric    = "numeric",
    integer    = "integer",
    logical    = "logical",
    stop("Invalid object or type")
  )
}

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
    logical    = c( length( obj ), 1L ),
    stop( "Invalid object" )
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
    logical    = 0L,
    stop( "Invalid object" )
  )
}

# Clean global env (and all memory)
clean.global <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}
