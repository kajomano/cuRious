# .Calls: src/transfer.cpp, src/tensor.cpp

# TODO ====
# Create from-to continous (range) subsetting

# TODO ====
# All checks should be in transfer, not in trnsfr.ptr

# This function is a general tool for copying data between tensors or R objects
# residing on any level, therefore it is horribly bloated :D
transfer <- function( src,
                      dst      = NULL,
                      cols.src = NULL,
                      cols.dst = NULL,
                      stream   = NULL ){

  # Src attributes
  type.src  <- get.type( src )
  dims.src  <- get.dims( src )
  level.src <- get.level( src )
  obj.src   <- get.obj( src )

  # Dst must exist
  if( is.null(dst) ){
    # If not, create an L0 object of the same type
    dst <- create.obj( dims.src, type = type.src )
  }

  # Dst attributes
  type.dst  <- get.type( dst )
  dims.dst  <- get.dims( dst )
  level.dst <- get.level( dst )
  obj.dst   <- get.obj( dst )

  # Type matching
  if( type.src != type.dst ){
    stop( "Object types do not match" )
  }

  # Source columns
  obj.cols.src <- NULL
  if( !is.null(cols.src) ){
    if( get.type( cols.src ) != "integer" ){
      stop( "Source columns are not integers" )
    }

    if( level.src < 3 ){
      if( level.dst == 3 ){
        stop( "No subsetting allowed between host and device transfers" )
      }

      if( get.level( cols.src ) == 3 ){
        stop( "Source columns are not in the host memory" )
      }
    }else{
      if( level.dst < 3 ){
        stop( "No subsetting allowed between host and device transfers" )
      }

      if( get.level( cols.src ) < 3 ){
        stop( "Source columns are not in the device memory" )
      }
    }

    obj.cols.src  <- get.obj( cols.src )
    dims.src[[2]] <- get.dims( cols.src )[[1]]
  }

  # Destination columns
  obj.cols.dst <- NULL
  if( !is.null(cols.dst) ){
    if( get.type( cols.dst ) != "integer" ){
      stop( "Destination columns are not integers" )
    }

    if( level.dst < 3 ){
      if( level.src == 3 ){
        stop( "No subsetting allowed between host and device transfers" )
      }

      if( get.level( cols.dst ) == 3 ){
        stop( "Destination columns are not in the host memory" )
      }
    }else{
      if( level.src < 3 ){
        stop( "No subsetting allowed between host and device transfers" )
      }

      if( get.level( cols.dst ) < 3 ){
        stop( "Destination columns are not in the device memory" )
      }
    }

    obj.cols.dst  <- get.obj( cols.dst )
    dims.dst[[2]] <- get.dims( cols.dst )[[1]]
  }

  # Dimension matching
  if( !identical( dims.src, dims.dst ) ){
    stop( "Dimensions do not match" )
  }

  # Stream check
  if( !is.null(stream) ){
    check.cuda.stream( stream )
    stream <- stream$get.stream
  }

  # Main low level transfer call
  transfer.obj( obj.src,
                obj.dst,
                level.src,
                level.dst,
                type.src,
                dims.src,
                obj.cols.src,
                obj.cols.dst,
                stream )

  invisible( dst )
}

# Low level transfer call that handles objects, for speed considerations
# no argument checks are done, don't use interactively or in any place where
# speed is not critical!
# Switch hell
transfer.obj = function( src,
                         dst,
                         level.src,
                         level.dst,
                         type,
                         dims,
                         cols.src = NULL,
                         cols.dst = NULL,
                         stream   = NULL ){
  res <- switch(
    paste0( get.level(src), get.level(dst), type ),
    # These will be doubles actually
    `00f` = .Call( "cuR_transf_0_0_f", src, dst, dims, cols.src, cols.dst ),
    `00i` = .Call( "cuR_transf_0_0_i", src, dst, dims, cols.src, cols.dst ),
    # Logicals are stored as integers
    `00b` = .Call( "cuR_transf_0_0_i", src, dst, dims, cols.src, cols.dst ),

    # ITT
    # ...

    `01` = .Call( "cuR_transf_0_12", src, dst, dims, cols.src, cols.dst ),
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
