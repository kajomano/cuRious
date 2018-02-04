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
  off.cols.src <- NULL
  obj.cols.src <- NULL

  if( !is.null( cols.src ) ){
    if( is.list( cols.src ) ){
      range.cols.src <- as.integer( c( cols.src[[1]], cols.src[[2]] ) )
      if( range.cols.src[[2]] > dims.src[[2]] ||
          range.cols.src[[2]] < range.cols.src[[1]] ||
          range.cols.src[[1]] < 0L ){
        stop( "Source column range out of bounds" )
      }
      off.cols.src  <- range.cols.src[[1]]
      dims.src[[2]] <- range.cols.src[[2]] - range.cols.src[[1]] + 1L
    }else{
      if( level.src == 3 || level.dst == 3 || get.level( cols.src ) == 3 ){
        stop( "Column subset does not work with L3 tensors" )
      }

      if( get.type( cols.src ) != "i" ){
        stop( "Source column subset is not integer" )
      }

      obj.cols.src  <- get.obj( cols.src )
      dims.src[[2]] <- get.dims( cols.src )[[1]]
    }
  }

  # Destination columns
  off.cols.dst <- NULL
  obj.cols.dst <- NULL

  if( !is.null( cols.dst ) ){
    if( is.list( cols.dst ) ){
      range.cols.dst <- as.integer( c( cols.dst[[1]], cols.dst[[2]] ) )
      if( range.cols.dst[[2]] > dims.dst[[2]] ||
          range.cols.dst[[2]] < range.cols.dst[[1]] ||
          range.cols.dst[[1]] < 0L ){
        stop( "Source column range out of bounds" )
      }
      off.cols.dst  <- range.cols.dst[[1]]
      dims.dst[[2]] <- range.cols.dst[[2]] - range.cols.dst[[1]] + 1L
    }else{
      if( level.src == 3 || level.dst == 3 || get.level( cols.dst ) == 3 ){
        stop( "Column subset does not work with L3 tensors" )
      }

      if( get.type( cols.dst ) != "i" ){
        stop( "Destiantio column subset is not integer" )
      }

      obj.cols.dst  <- get.obj( cols.dst )
      dims.dst[[2]] <- get.dims( cols.dst )[[1]]
    }
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
  transfer.core( obj.src,
                 obj.dst,
                 level.src,
                 level.dst,
                 type.src,
                 dims.src,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst,
                 stream )

  # Return destination
  invisible( dst )
}

# Low level transfer call that handles objects, for speed considerations
# no argument checks are done, don't use interactively or in any place where
# speed is not critical!
# Switch hell
transfer.core = function( src,
                         dst,
                         level.src,
                         level.dst,
                         type,
                         dims,
                         off.cols.src = NULL,
                         off.cols.dst = NULL,
                         obj.cols.src = NULL,
                         obj.cols.dst = NULL,
                         stream       = NULL ){
  res <- switch(
    get.level(src) + 1,
    `0` = {
      switch(
        get.level(dst) + 1,
        `0` = {
          .Call( paste0( "cuR_transfer_0_0_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        }
      )
    }
  )

  # res <- switch(
  #   paste0( get.level(src), get.level(dst), type ),
  #   `00n` = .Call( "cuR_transfer_0_0_n", src, dst, dims, off.cols.src, off.cols.dst, obj.cols.src, obj.cols.dst ),
  #   `00i` = .Call( "cuR_transfer_0_0_i", src, dst, dims, off.cols.src, off.cols.dst, obj.cols.src, obj.cols.dst ),
  #   `00l` = .Call( "cuR_transfer_0_0_l", src, dst, dims, off.cols.src, off.cols.dst, obj.cols.src, obj.cols.dst )

    # ITT
    # ...

    # `01` = .Call( "cuR_transf_0_12", src, dst, dims, cols.src, cols.dst ),
    # `02` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_0_12", src, dst, dims, cols.src, cols.dst )
    # },
    # `03` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   # This is a multi-stage transfer
    #   temp <- create.dummy( dims.dst, 2L )
    #   trnsfr.ptr( src, temp, cols.src, cols.dst )
    #   trnsfr.ptr( temp, dst )
    #   .Call( "cuR_destroy_tensor_2", temp )
    # },
    #
    # `10` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_0", src, dst, dims, cols.src, cols.dst )
    # },
    # `11` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    # },
    # `12` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    # },
    # `13` = {
    #   dims <- check.dims( dims.src, dims.dst )
    #   .Call( "cuR_transf_1_3", src, dst, dims )
    # },
    #
    # `20` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_0", src, dst, dims, cols.src, cols.dst )
    # },
    # `21` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    # },
    # `22` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   .Call( "cuR_transf_12_12", src, dst, dims, cols.src, cols.dst )
    # },
    # `23` = {
    #   dims <- check.dims( dims.src, dims.dst )
    #   .Call( "cuR_transf_2_3", src, dst, dims, stream )
    # },
    #
    # `30` = {
    #   dims <- check.dims( dims.src, dims.dst, cols.src, cols.dst )
    #   # This is a multi-stage transfer
    #   temp <- create.dummy( dims.src, 2L )
    #   trnsfr.ptr( src, temp )
    #   trnsfr.ptr( temp, dst, cols.src, cols.dst )
    #   .Call( "cuR_destroy_tensor_2", temp )
    # },
    # `31` = {
    #   dims <- check.dims( dims.src, dims.dst )
    #   .Call( "cuR_transf_3_1", src, dst, dims )
    # },
    # `32` = {
    #   dims <- check.dims( dims.src, dims.dst )
    #   .Call( "cuR_transf_3_2", src, dst, dims, stream )
    # },
    # `33` = {
    #   dims <- check.dims( dims.src, dims.dst )
    #   .Call( "cuR_transf_3_3", src, dst, dims, stream )
    # }
  # )

  if( is.null(res) ){
    stop( "Transfer was unsuccessful" )
  }
}
