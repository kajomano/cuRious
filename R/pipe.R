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

      if( !is.null( src.perm ) ){
        src.dims$check.perm( src.perm )
      }

      if( !is.null( dst.perm ) ){
        dst.dims$check.perm( dst.perm )
      }

      if( !is.null( src.span ) ){
        src.dims$check.span( src.span )
      }

      if( !is.null( dst.span ) ){
        dst.dims$check.span( dst.span )
      }

      if( !identical( src.dims$dims, dst.dims$dims ) ){
        stop( "Dimensions do not match" )
      }

      # Assignments
      private$.add.ep( src, "src" )
      private$.add.ep( dst, "dst", TRUE )

      private$.params$type <- src$type
      private$.params$dims <- src.dims$dims

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
    .update.context = function(){
      # Since levels are the primary dynamically changing attribute of tensors,
      # these checks mostly concern them
      src <- private$.eps$src
      dst <- private$.eps$dst

      private$.params$src.level <- src$level
      private$.params$dst.level <- dst$level

      deep.transf  <- ( ( src$level == 3L ) && ( dst$level == 3L ) )

      cross.transf <- ( ( ( src$level != 3L ) && ( dst$level == 3L ) ) ||
                        ( ( dst$level != 3L ) && ( src$level == 3L ) ) ||
                        ( deep.transf && ( src$device != dst$device ) ) )

      src.perm <- private$.eps$src.perm
      dst.perm <- private$.eps$dst.perm

      if( !is.null( src.perm ) ){
        if( cross.transf ){
          stop( "Source permutation is not available between these levels" )
        }

        if( ( deep.transf  && ( src.perm$level != 3L ) ) ||
            ( !deep.transf && ( src.perm$level == 3L ) ) ){
          stop( "Source permutation tensor is not on the correct level" )
        }

        if( deep.transf && ( src$device != src.perm$device ) ){
          stop( "Source permutation is not on the correct device" )
        }
      }

      if( !is.null( dst.perm ) ){
        if( cross.transf ){
          stop( "Destination permutation is not available between these levels" )
        }

        if( ( deep.transf  && ( dst.perm$level != 3L ) ) ||
            ( !deep.transf && ( dst.perm$level == 3L ) ) ){
          stop( "Destination permutation tensor is not on the correct level" )
        }

        if( deep.transf && ( src$device != dst.perm$device ) ){
          stop( "Destination permutation is not on the correct device" )
        }
      }

      if( !is.null( private$.eps$stream ) ){
        if( private$.eps$stream$is.active ){
          if( ( src$level %in% c( 0L, 1L ) ) || ( dst$level %in% c( 0L, 1L ) ) ){
            stop( "An active stream is given to a synchronous transfer" )
          }

          if( deep.transf ){
            if( src$device != dst$device ){
              stop( "An active stream is given to a synchronous transfer" )
            }

            if( private$.eps$stream$device != src$device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }
      }

      # This only works because you dont have to set the device correctly
      # if no kernels are run, and kernels are only run on L3-L3 same device
      # transfers
      private$.device <- src$device

      # Multi or single-step transfer
      if( ( ( src$level == 0L ) && ( dst$level == 3L ) ) ||
          ( ( src$level == 3L ) && ( dst$level == 0L ) ) ||
          ( deep.transf && ( src$device != dst$device ) ) ){
        private$.fun <- .transfer.ptr.multi
      }else{
        private$.fun <- .transfer.ptr
      }

      super$.update.context()
    }
  )
)
