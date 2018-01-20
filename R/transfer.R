# .Calls: src/transfer.cpp, src/tensor.cpp
#
# This function is a general tool for copying data between tensors or R objects
# residing on any level, therefore it is horribly bloated
transfer <- function( src,
                      dst = NULL,
                      cols.src = NULL,
                      cols.dst = NULL,
                      stream = NULL,
                      threads = 4L ){

  # Solidify destination, levels, dims and objects
  dims.src <- get.dims( src )
  if( is.tensor(src) ){
    src.obj <- src$access
  }else{
    src.obj <- src
  }

  if( is.null(dst) ){
    dims.dst <- dims.src
    dst.obj <- create.dummy( dims.dst )
  }else{
    dims.dst <- get.dims( dst )
    if( is.tensor(dst) ){
      dst.obj <- dst$access
    }else{
      dst.obj <- dst
    }
  }

  src.level <- get.level( src )
  dst.level <- get.level( dst )

  # Clean up other arguments
  if( !is.null(cols.src) ) cols.src <- force.int( cols.src )
  if( !is.null(cols.dst) ) cols.dst <- force.int( cols.dst )
  threads <- force.int( threads )
  if( !is.null(stream) ) check.cuda.stream( stream )

  # Single-stage transfers (subsettable, asyncable or none)
  if( abs( src.level - dst.level ) != 3L ){
    # Subsettable transfers host-host
    if( src.level < 3L && dst.level < 3L ){
      if( !is.null( cols.src ) ) dims.src[[2]] <- length( cols.src )
      if( !is.null( cols.dst ) ) dims.dst[[2]] <- length( cols.dst )
      check.dims( dims.src, dims.dst )

      if( src.level == 0L && dst.level == 0L ){
        # Level 0 copy+subset
        .Call( "cuR_transf_0_0", src.obj, dst.obj, dims.src, cols.src, cols.dst )
      }else if( src.level != 0L && dst.level != 0L ){
        # Level 1/2 copy+subset
        .Call( "cuR_transf_12_12", src.obj, dst.obj, dims.src, cols.src, cols.dst )
      }else if( src.level == 0L ){
        # Subset+convert to float
        .Call( "cuR_transf_0_12", src.obj, dst.obj, dims.src, cols.src, cols.dst, threads )
      }else{
        # Subset+convert to double
        .Call( "cuR_transf_12_0", src.obj, dst.obj, dims.src, cols.src, cols.dst, threads )
      }

      # Non-async host-device, device-host
    }else if( src.level == 1L || dst.level == 1L ){
      check.dims( dims.src, dims.dst )

      if( src.level == 1L ){
        # TODO ====
      }else{
        # TODO ====
      }

      # Asyncable transfers host-device, device-device, device-host
    }else{
      check.dims( dims.src, dims.dst )

      if( src.level == 2L && dst.level == 3L ){
        # TODO ====
      }else if( src.level == 3L && dst.level == 2L ){
        # TODO ====
      }else{
        # TODO ====
      }
    }
    # Multi-stage transfers (subsettable)
  }else{
    if( !is.null( cols.src ) ) dims.src[[2]] <- length( cols.src )
    if( !is.null( cols.dst ) ) dims.dst[[2]] <- length( cols.dst )
    check.dims( dims.src, dims.dst )

    if( src.level == 0L ){
      # TODO ====
    }else{
      # TODO ====
    }
  }

  # Return with dst if wasnt supported
  if( is.null(dst) ){
    return( dst.obj )
  }

  invisible( TRUE )
}

check.dims = function( dims1, dims2 ){
  if( !identical( dims1, dims2 ) ){
    stop( "Dimensions do not match" )
  }
}
