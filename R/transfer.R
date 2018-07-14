# .Calls: src/transfer.cpp
# Highly connected to code in pipe.R

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

# Low level transfer calls that handles ptrs, for speed considerations
# no argument checks are done, don't use interactively!
.transfer.ptr.choose = function( src.level,
                                 dst.level,
                                 src.device = NULL,
                                 dst.device = NULL ){

  deep <- ( src.level == 3L && dst.level == 3L )
  if( deep ){
    if( is.null( src.device ) || is.null( dst.device ) ){
      stop( "Missing devices for transfer call choice" )
    }
  }

  if( ( src.level == 3L && dst.level == 0L ) ||
      ( src.level == 0L && dst.level == 3L ) ||
      deep && ( src.device != dst.device ) ){
    .transfer.ptr.multi
  }else{
    .transfer.ptr.uni
  }
}

# Single-step transfer calls
transfer.ptr = function( src.tensor,
                          dst.tensor,
                          src.level,
                          dst.level,
                          type,
                          dims,
                          src.dims        = NULL,
                          dst.dims        = NULL,
                          src.perm.tensor = NULL,
                          dst.perm.tensor = NULL,
                          src.span.off    = NULL,
                          dst.span.off    = NULL,
                          context.workers = NULL,
                          stream.queue    = NULL ){

  .Call( "cuR_transfer",
         src.tensor,
         dst.tensor,
         src.level,
         dst.level,
         type,
         dims,
         src.dims,
         dst.dims,
         src.perm.tensor,
         dst.perm.tensor,
         src.span.off,
         dst.span.off,
         context.workers,
         stream.queue )
}

# # Multi-transfer calls: 0L-2L-3L, 3L-2L-0L or 3L-2L-3L on different devices
# # Context and stream is ignored
# .transfer.ptr.multi = function( src.tensor,
#                                 dst.tensor,
#                                 src.level,
#                                 dst.level,
#                                 type,
#                                 dims,
#                                 src.dims        = NULL,
#                                 dst.dims        = NULL,
#                                 src.perm.tensor = NULL,
#                                 dst.perm.tensor = NULL,
#                                 src.span.off    = NULL,
#                                 dst.span.off    = NULL,
#                                 context.workers = NULL,
#                                 stream.queue    = NULL ){
#
#   # Multi-transfer call 0L-2L-3L or 3L-2L-0L
#   tmp.tensor <- .Call( "cuR_tensor_create", 2L, dims, type )
#
#   .Call( "cuR_transfer",
#          src.tensor,
#          tmp.tensor,
#          src.level,
#          2L,
#          type,
#          dims,
#          src.dims,
#          dims,
#          src.perm.tensor,
#          NULL,
#          src.span.off,
#          NULL,
#          NULL,
#          NULL )
#
#   .Call( "cuR_transfer",
#          tmp.tensor,
#          dst.tensor,
#          2L,
#          dst.level,
#          type,
#          dims,
#          dims,
#          dst.dims,
#          NULL,
#          dst.perm.tensor,
#          NULL,
#          dst.span.off,
#          NULL,
#          NULL )
#
#   .Call( "cuR_tensor_destroy", tmp.tensor, 2L, type )
# }
