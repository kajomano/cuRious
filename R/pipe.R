# Pipes are for transferring data between tensors. Pipes do most of the
# argument sanity checks at creation, and try to do the rest only when needed at
# runtime. Overhead reduction is key for smaller tasks.

pipe <- R6Class(
  "cuR.pipe",
  inherit = fusion,
  public = list(
    initialize = function( src,
                           dst,
                           src.perm = NULL,
                           dst.perm = NULL,
                           src.span = NULL,
                           dst.span = NULL,
                           stream   = NULL  ){
      # Sanity checks
      check.tensor( src )
      check.tensor( dst )

      if( src$type != dst$type ){
        stop( "Tensor types do not match" )
      }

      # Dim checks
      src.dims <- .tensor.dims$new( src )
      dst.dims <- .tensor.dims$new( dst )

      src.dims$check.perm( src.perm )
      dst.dims$check.perm( dst.perm )
      src.dims$check.span( src.span )
      dst.dims$check.span( dst.span )

      if( !identical( src.dims$dims, dst.dims$dims ) ){
        stop( "Dimensions do not match" )
      }

      # Assignments
      private$.add.ep( src, "src" )
      private$.add.ep( dst, "dst", TRUE )

      private$.params$type         <- src$type
      private$.params$src.dims     <- src.dims$dims.orig
      private$.params$dst.dims     <- src.dims$dims.orig
      private$.params$dims         <- src.dims$dims

      private$.params$src.span.off <- src.dims$span.off
      private$.params$dst.span.off <- dst.dims$span.off

      private$.add.ep( src.perm, "src.perm" )
      private$.add.ep( dst.perm, "dst.perm" )

      if( !is.null( stream ) ){
        check.cuda.stream( stream )
      }

      private$.add.ep( stream, "stream" )
    }
  ),

  private = list(
    .update.context = function( ... ){
      # Since levels are the primary dynamically changing attribute of tensors,
      # these checks mostly concern them

      # Temporary variables: $lookups take a long time, it makes sense to save
      # accessed values if used multiple times
      src           <- private$.eps$src
      dst           <- private$.eps$dst
      src.level     <- src$level
      dst.level     <- dst$level
      src.device    <- src$device
      dst.device    <- dst$device

      src.perm      <- private$.eps$src.perm
      dst.perm      <- private$.eps$dst.perm

      transfer.deep <- ( src.level == 3L && dst.level == 3L )

      if( !is.null( src.perm ) ){
        src.perm.level  <- src.perm$level

        if( transfer.deep ){
          if( src.device == dst.device ){
            if( src.perm.level != 3L ){
              stop( "Source permutation tensor is not on the correct level" )
            }

            if( src.perm$device != src.device ){
              stop( "Source permutation tensor is not on the correct device" )
            }
          }else{
            if( src.perm.level == 3L ){
              stop( "Source permutation tensor is not on the correct level" )
            }
          }
        }else{
          if( src.perm.level == 3L ){
            stop( "Source permutation tensor is not on the correct level" )
          }
        }
      }

      if( !is.null( dst.perm ) ){
        dst.perm.level  <- dst.perm$level

        if( transfer.deep ){
          if( src.device == dst.device ){
            if( dst.perm.level != 3L ){
              stop( "Destination permutation tensor is not on the correct level" )
            }

            if( dst.perm$device != dst.device ){
              stop( "Destination permutation tensor is not on the correct device" )
            }
          }else{
            if( dst.perm.level == 3L ){
              stop( "Destination permutation tensor is not on the correct level" )
            }
          }
        }else{
          if( dst.perm.level == 3L ){
            stop( "Destination permutation tensor is not on the correct level" )
          }
        }
      }

      if( !is.null( private$.eps$stream ) ){
        stream <- private$.eps$stream

        if( stream$is.active ){
          if( transfer.deep ){
            if( stream$device != src.device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }
      }

      # This only works because you dont have to set the device correctly
      # if no kernels are run, and kernels are only run on L3-L3 same device
      # transfers
      private$.device           <- src.device
      private$.params$src.level <- src.level
      private$.params$dst.level <- dst.level

      # Multi or single-step transfer
      if( ( src.level == 0L && dst.level == 3L ) ||
          ( src.level == 3L && dst.level == 0L ) ||
          ( transfer.deep && ( src.device != dst.device ) ) ){
        private$.fun <- .transfer.ptr.multi
      }else{
        private$.fun <- .transfer.ptr.uni
      }

      print( "context updated" )
    }
  )
)
