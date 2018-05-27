# .Calls: src/transfer.cpp
# Highly connected to code in tunnel.R

# High level transfer call. Thin wrapper around a temporary pipe. Should only
# be used in non speed-critical places.
transfer <- function( src,
                      dst,
                      src.perm = NULL,
                      dst.perm = NULL,
                      src.span = NULL,
                      dst.span = NULL,
                      stream   = NULL ){

  pip <- pipe$new( src, dst, src.perm, dst.perm, src.span, dst.span, stream )
  res <- pip$run()
  pip$destroy()
  invisible( res )
}

# Low level transfer call that handles ptrs, for speed considerations
# no argument checks are done, don't use interactively!
# Switch hell
.transfer.ptr = function( src.ptr,
                          dst.ptr,
                          src.level,
                          dst.level,
                          type,
                          src.dims,
                          dst.dims,
                          dims,
                          src.perm.ptr = NULL,
                          dst.perm.ptr = NULL,
                          src.span.off = NULL,
                          dst.span.off = NULL,
                          stream.ptr   = NULL ){
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
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_0_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
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
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_1_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.span.off,
                 dst.span.off )
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
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `1` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_12_12_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_2_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.span.off,
                 dst.span.off,
                 stream.ptr )
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
                 src.span.off,
                 dst.span.off )
        },
        `2` = {
          .Call( paste0( "cuR_transfer_3_2_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.span.off,
                 dst.span.off,
                 stream.ptr )
        },
        `3` = {
          .Call( paste0( "cuR_transfer_3_3_", type ),
                 src.ptr,
                 dst.ptr,
                 dims,
                 src.perm.ptr,
                 dst.perm.ptr,
                 src.span.off,
                 dst.span.off,
                 stream.ptr )
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

# Multi-transfer call 0L-2L-3L or 3L-2L-0L
.transfer.ptr.multi = function( src.ptr,
                                dst.ptr,
                                src.level,
                                dst.level,
                                type,
                                dims,
                                src.perm.ptr = NULL,
                                dst.perm.ptr = NULL,
                                src.span.off = NULL,
                                dst.span.off = NULL,
                                stream.ptr   = NULL ){

  # Multi-transfer call 0L-2L-3L or 3L-2L-0L
  tmp <- tensor$new( NULL, 2L, dims, type )

  .transfer.ptr( src.ptr,
                 tmp$ptr,
                 src.level,
                 2L,
                 type,
                 dims,
                 NULL,
                 NULL,
                 src.span.off )

  .transfer.ptr( tmp$ptr,
                 dst.ptr,
                 2L,
                 dst.level,
                 type,
                 dims,
                 NULL,
                 NULL,
                 NULL,
                 dst.span.off )

  tmp$destroy()

  invisible( TRUE )
}
