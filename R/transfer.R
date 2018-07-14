# .Calls: src/transfer.cpp
# Highly connected to code in pipe.R

# High level transfer call, wrapper around temporary pipe(s). Should only
# be used in non speed-critical places.
transfer <- function( src,
                      dst,
                      src.perm = NULL,
                      dst.perm = NULL,
                      src.span = NULL,
                      dst.span = NULL ){

  src.level <- src$level
  dst.level <- dst$level

  deep.transfer <- ( src.level == 3L && dst$level == 3L )
  if( ( src.level == 3L && dst.level == 0L ) ||
      ( src.level == 0L && dst.level == 3L ) ||
      deep.transfer && ( src$device != dst$device ) ){
    # Multi-step transfers

    src.dims <- .tensor.dims$new( src )
    dst.dims <- .tensor.dims$new( dst )

    src.dims$check.perm( src.perm )
    dst.dims$check.perm( dst.perm )
    src.dims$check.span( src.span )
    dst.dims$check.span( dst.span )

    if( !identical( src.dims$dims, dst.dims$dims ) ){
      stop( "Dimensions do not match" )
    }

    tmp  <- tensor$new( NULL, 2L, src.dims$dims, src$type )
    pip1 <- pipe$new( src, tmp, src.perm, NULL, src.span, NULL )
    pip2 <- pipe$new( tmp, dst, NULL, dst.perm, NULL, dst.span )

    pip1$run()
    pip2$run()

    tmp$destroy()
    pip1$destroy()
    pip2$destroy()
  }else{
    # Single-step transfers

    pip <- pipe$new( src, dst, src.perm, dst.perm, src.span, dst.span )
    pip$run()
    pip$destroy()
  }

  invisible( TRUE )
}
