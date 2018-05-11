# .Calls: src/transfer.cpp

# This function is a general tool for copying data between tensors
# residing on any level. cols.dst and .src are either 2 long integer vectors
# defining column ranges, or n long integer tensors on levels 0-2, defining
# exact columns to be transferred (transferred to).
transfer <- function( src,
                      dst,
                      src.cols = NULL,
                      dst.cols = NULL,
                      stream   = NULL ){

  src <- check.tensor( src )
  dst <- check.tensor( dst )

  # Type matching
  if( src$type != dst$type ){
    stop( "Tensor types do not match" )
  }

  # Source column subset/ranges
  if( !is.null( src.cols ) ){
    if( is.tensor( src.cols ) ){
      src.subs <- column.indiv( src, src.cols )
    }else if( is.obj( src.cols ) ){
      src.subs <- column.range( src, src.cols )
    }else{
      stop( "Invalid source column subset" )
    }
  }else{
    src.subs <- column.empty( src, src.cols )
  }

  # Destination column subset/ranges
  if( !is.null( dst.cols ) ){
    if( is.tensor( dst.cols ) ){
      dst.subs <- column.indiv( dst, dst.cols )
    }else if( is.obj( dst.cols ) ){
      dst.subs <- column.range( dst, dst.cols )
    }else{
      stop( "Invalid destination column subset" )
    }
  }else{
    dst.subs <- column.empty( dst, dst.cols )
  }

  # Dimension matching
  if( !identical( src.subs$dims, dst.subs$dims ) ){
    stop( "Dimensions do not match" )
  }

  # Stream check
  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$stream
  }

  # Main core transfer call
  transfer.core( src$ptr,
                 dst$ptr,
                 src$level,
                 dst$level,
                 src$type,
                 src.subs$dims,
                 src.subs$off,
                 dst.subs$off,
                 src.subs$ptr,
                 dst.subs$ptr,
                 stream )

  invisible( TRUE )
}

# Mid-level transfer call, without argument checks. Can still handle
# multi-transfer calls. Should not be used interactively!
transfer.core = function( src.ptr,
                          dst.ptr,
                          src.level,
                          dst.level,
                          type,
                          dims,
                          src.col.off = NULL,
                          dst.col.off = NULL,
                          src.col.ptr = NULL,
                          dst.col.ptr = NULL,
                          stream      = NULL ){

  # Main low level transfer calls
  if( src.level == 0L && dst.level == 3L ||
      src.level == 3L && dst.level == 0L ){

    # Multi-transfer call 0L-2L-3L or 3L-2L-0L
    tmp <- tensor$new( NULL, 2L, dims, type )

    transfer.ptr( src.ptr,
                  tmp$ptr,
                  src.level,
                  2L,
                  type,
                  dims,
                  src.col.off,
                  NULL )

    transfer.ptr( tmp$ptr,
                  dst.ptr,
                  2L,
                  dst.level,
                  type,
                  dims,
                  NULL,
                  dst.col.off )

    tmp$destroy()
  }else{
    # Single-transfer calls
    transfer.ptr( src.ptr,
                  dst.ptr,
                  src.level,
                  dst.level,
                  type,
                  dims,
                  src.col.off,
                  dst.col.off,
                  src.col.ptr,
                  dst.col.ptr,
                  stream )
  }

  invisible( TRUE )
}

# Low level transfer call that handles objects, for speed considerations
# no argument checks are done, don't use interactively or in any place where
# speed is not critical!
# Switch hell
transfer.ptr = function( src.ptr,
                         dst.ptr,
                         src.level,
                         dst.level,
                         type,
                         dims,
                         src.col.off = NULL,
                         dst.col.off = NULL,
                         src.col.ptr = NULL,
                         dst.col.ptr = NULL,
                         stream      = NULL ){
  res <- switch(
    as.character( src.level ),
    `0` = {
      switch(
        as.character( dst.level ),
        `0` = {
          .Call( paste0( "cuR_transfer_0_0_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        stop( "Invalid level" )
      )
    },
    `1` = {
      switch(
        as.character( dst.level ),
        `0` = {
          .Call( paste0( "cuR_transfer_12_0_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_1_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off )
        },
        stop( "Invalid level" )
      )
    },
    `2` = {
      switch(
        as.character( dst.level ),
        `0` = {
          .Call( paste0( "cuR_transfer_12_0_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 src.col.ptr,
                 dst.col.ptr )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_2_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 stream )
        },
        stop( "Invalid level" )
      )
    },
    `3` = {
      switch(
        as.character( dst.level ),
        `1` = {
          .Call( paste0( "cuR_transfer_3_1_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_3_2_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
                 stream )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_3_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.col.off,
                 dst.col.off,
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

  invisible( TRUE )
}
