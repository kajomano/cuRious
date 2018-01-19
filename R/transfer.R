# .Calls: src/transfer.cpp, src/tensor.cpp
#
# This function is a general tool for copying data between tensors or R objects
# residing on any level, therefore it is horribly bloated
transfer <- function( src,
                      dst = NULL,
                      rows.src = NULL,
                      rows.dst = NULL,
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
  if( !is.null(rows.src) ) rows.src <- force.int( rows.src )
  if( !is.null(rows.dst) ) rows.dst <- force.int( rows.dst )
  threads <- force.int( threads )
  if( !is.null(stream) ) check.cuda.stream( stream )

  # Single-stage transfers (subsettable, asyncable or none)
  if( abs( src.level - dst.level ) != 3L ){
    # Subsettable transfers host-host
    if( src.level < 3L && dst.level < 3L ){
      if( !is.null( stream ) ) warning( "Stream supported to a non-async transfer" )
      if( !is.null( rows.src ) ) dims.src[[1]] <- length( rows.src )
      if( !is.null( rows.dst ) ) dims.dst[[1]] <- length( rows.dst )
      check.dims( dims.src, dims.dst )

      if( src.level == 0L && dst.level == 0L ){
        # TODO ====
        # Level 0 copy+subset
        .Call( "cuR_transf_0_0", src.obj, dst.obj, dims.src, rows.src, rows.dst, threads )
      }else if( src.level != 0L && dst.level != 0L ){
        # TODO ====
        # Level 1/2 copy+subset
      }else if( src.level == 0L ){
        # TODO ====
        # Subset+convert to float
      }else{
        # TODO ====
        # Subset+convert to double
      }

      # Non-async host-device, device-host
    }else if( src.level == 1L || dst.level == 1L ){
      if( !is.null( stream ) ) warning( "Stream supported to a non-async transfer" )
      if( !is.null( rows.src ) || !is.null( rows.dst ) ) warning( "Row subset supported to a non-subsettable transfer" )
      check.dims( dims.src, dims.dst )

      if( src.level == 1L ){
        # TODO ====
      }else{
        # TODO ====
      }

      # Asyncable transfers host-device, device-device, device-host
    }else{
      if( !is.null( rows.src ) || !is.null( rows.dst ) ) warning( "Row subset supported to a non-subsettable transfer" )
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
    if( !is.null( stream ) ) warning( "Stream supported to a non-async transfer" )
    if( !is.null( rows.src ) ) dims.src[[1]] <- length( rows.src )
    if( !is.null( rows.dst ) ) dims.dst[[1]] <- length( rows.dst )
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
