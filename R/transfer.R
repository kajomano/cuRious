# .Calls: src/transfer.cpp

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

  # Source column subset/ranges
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

  # Destination column subset/ranges
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
        stop( "Destiantion column subset is not integer" )
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

  # Main low level transfer calls
  if( level.src == 0L && level.dst == 3L || level.src == 3L && level.dst == 0L ){
    # Multi-transfer call 0L-2L-3L or 3L-2L-0L
    temp <- create.obj( dims.src, 2, type.src )

    transfer.core( obj.src,
                   temp,
                   level.src,
                   2L,
                   type.src,
                   dims.src,
                   off.cols.src,
                   NULL )

    transfer.core( temp,
                   obj.dst,
                   2L,
                   level.dst,
                   type.src,
                   dims.src,
                   NULL,
                   off.cols.dst )

    destroy.obj( temp )
  }else{
    # Single-transfer calls
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
  }

  # Return destination
  dst
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
    as.character(get.level(src)),
    `0` = {
      switch(
        as.character(get.level(dst)),
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
        },
        stop( "Invalid level" )
      )
    },
    `1` = {
      switch(
        as.character(get.level(dst)),
        `0` = {
          .Call( paste0( "cuR_transfer_12_0_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_1_3_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst )
        },
        stop( "Invalid level" )
      )
    },
    `2` = {
      switch(
        as.character(get.level(dst)),
        `0` = {
          .Call( paste0( "cuR_transfer_12_0_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 obj.cols.src,
                 obj.cols.dst )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_2_3_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 stream )
        },
        stop( "Invalid level" )
      )
    },
    `3` = {
      switch(
        as.character(get.level(dst)),
        `1` = {
          .Call( paste0( "cuR_transfer_3_1_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_3_2_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 stream )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_3_3_", type ),
                 src,
                 dst,
                 dims,
                 off.cols.src,
                 off.cols.dst,
                 stream )
        },
        stop( "Invalid level" )
      )
    },
    stop( "Invalid level" )
  )

  if( is.null(res) ){
    stop( "Transfer was unsuccessful" )
  }
}
