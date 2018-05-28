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

  if( ( src.level == 3L && dst.level == 0L ) ||
      ( src.level == 0L && dst.level == 3L ) ){
    .transfer.ptr.multi( src.ptr,
                         dst.ptr,
                         src.level,
                         dst.level,
                         type,
                         src.dims,
                         dst.dims,
                         dims,
                         src.perm.ptr,
                         dst.perm.ptr,
                         src.span.off,
                         dst.span.off,
                         stream.ptr )
  }else{
    .transfer.ptr.uni( src.ptr,
                       dst.ptr,
                       src.level,
                       dst.level,
                       type,
                       src.dims,
                       dst.dims,
                       dims,
                       src.perm.ptr,
                       dst.perm.ptr,
                       src.span.off,
                       dst.span.off,
                       stream.ptr )
  }
}

.transfer.ptr.uni = function( src.ptr,
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

  .Call( "cuR_transfer",
         src.ptr,
         dst.ptr,
         src.level,
         dst.level,
         type,
         src.dims,
         dst.dims,
         dims,
         src.perm.ptr,
         dst.perm.ptr,
         src.span.off,
         dst.span.off,
         stream.ptr )
}

# Multi-transfer call 0L-2L-3L or 3L-2L-0L
.transfer.ptr.multi = function( src.ptr,
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

  # Multi-transfer call 0L-2L-3L or 3L-2L-0L
  tmp.ptr <- .Call( "cuR_tensor_create", 2L, dims, type )

  .Call( "cuR_transfer",
         src.ptr,
         tmp.ptr,
         src.level,
         2L,
         type,
         src.dims,
         dims,
         dims,
         src.perm.ptr,
         NULL,
         src.span.off,
         NULL,
         NULL )

  .Call( "cuR_transfer",
         tmp.ptr,
         dst.ptr,
         2L,
         dst.level,
         type,
         dims,
         dst.dims,
         dims,
         NULL,
         dst.perm.ptr,
         NULL,
         dst.span.off,
         NULL )

  .Call( "cuR_tensor_destroy", tmp.ptr, 2L, type )
}
