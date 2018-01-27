# .Calls: src/transfer.cpp, src/tensor.cpp

# TODO ====
# Create from-to continous subsetting

# This function is a general tool for copying data between tensors or R objects
# residing on any level, therefore it is horribly bloated
transfer <- function( src,
                      dst      = NULL,
                      cols.src = NULL,
                      cols.dst = NULL,
                      stream   = NULL ){

  # Clean up arguments
  # src
  dims.src <- get.dims( src )
  if( is.tensor(src) ){
    src.obj <- src$get.tensor
  }else{
    src.obj <- force.double( src )
  }

  # dst
  if( is.null(dst) ){
    dims.dst <- dims.src
    dst.obj <- create.dummy( dims.dst )
  }else{
    dims.dst <- get.dims( dst )
    if( is.tensor(dst) ){
      dst.obj <- dst$get.tensor
    }else{
      dst.obj <- force.double( dst )
    }
  }

  # cols.src
  if( !is.null(cols.src) ){
    cols.src <- force.int( cols.src )
    if( max( cols.src ) > dims.src[[2]] ){
      stop( "Col subset outside range" )
    }
  }

  # cols.dst
  if( !is.null(cols.dst) ){
    cols.dst <- force.int( cols.dst )
    if( max( cols.dst ) > dims.dst[[2]] ){
      stop( "Col subset outside range" )
    }
  }

  # stream
  if( !is.null(stream) ){
    check.cuda.stream( stream )
    stream <- stream$get.stream
  }

  # Main low level transfer call
  trnsfr.ptr( src.obj, dst.obj, cols.src, cols.dst, stream )

  # Return with dst if wasnt supported
  if( is.null(dst) ){
    return( dst.obj )
  }

  invisible( TRUE )
}

# Low level transfer call that handles tensor.ptr-s, for speed considerations
# no argument checks (except for dims) are done, don't use interactively!
# Switch hell
trnsfr.ptr = function( src,
                       dst,
                       cols.src = NULL,
                       cols.dst = NULL,
                       stream   = NULL ){

  dims.src <- get.dims( src )
  dims.dst <- get.dims( dst )

  res <- switch(
    paste0( get.level(src), get.level(dst) ),
    `00` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_0_0", src, dst, dims, cols.src, cols.dst )
    },
    `01` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_0_12", src, dst, dims, cols.src, cols.dst )
    },
    `02` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_0_12", src, dst, dims, cols.src, cols.dst )
    },
    `03` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      # This is a multi-stage transfer
      temp <- create.dummy( dims.dst, 2L )
      trnsfr.ptr( src, temp, cols.src, cols.dst )
      trnsfr.ptr( temp, dst )
      .Call( "cuR_destroy_tensor_2", temp )
    },

    `10` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_0", src, dst, dims, cols.src, cols.dst )
    },
    `11` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    },
    `12` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    },
    `13` = {
      dims <- check.dims( dims.src, dims.dst )
      .Call( "cuR_transf_1_3", src, dst, dims )
    },

    `20` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_0", src, dst, dims, cols.src, cols.dst )
    },
    `21` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    },
    `22` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    },
    `23` = {
      dims <- check.dims( dims.src, dims.dst )
      .Call( "cuR_transf_2_3", src, dst, dims, stream )
    },

    `30` = {
      dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
      # This is a multi-stage transfer
      temp <- create.dummy( dims.src, 2L )
      trnsfr.ptr( src, temp )
      trnsfr.ptr( temp, dst, cols.src, cols.dst )
      .Call( "cuR_destroy_tensor_2", temp )
    },
    `31` = {
      dims <- check.dims( dims.src, dims.dst )
      .Call( "cuR_transf_3_1", src, dst, dims )
    },
    `32` = {
      dims <- check.dims( dims.src, dims.dst )
      .Call( "cuR_transf_3_2", src, dst, dims, stream )
    },
    `33` = {
      dims <- check.dims( dims.src, dims.dst )
      .Call( "cuR_transf_3_3", src, dst, dims, stream )
    }
  )

  if( is.null(res) ){
    stop( "Transfer was unsuccessful" )
  }
}

# Helper functions ====
check.dims = function( dims1, dims2, cols1 = NULL, cols2 = NULL ){
  if( !is.null( cols1 ) ) dims1[[2]] <- length( cols1 )
  if( !is.null( cols2 ) ) dims2[[2]] <- length( cols2 )

  if( !identical( dims1, dims2 ) ){
    stop( "Dimensions do not match" )
  }

  dims1
}
