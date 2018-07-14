# Pipes are for transferring data between tensors. Pipes do most of the
# argument sanity checks at creation, and try to do the rest only when needed at
# runtime. Overhead reduction is key for smaller tasks.

# Pipe context class ====
pipe.context <- R6Class(
  "cuR.pipe.context",
  inherit = fusion.context,
  public = list(
    initialize = function( workers = 4L, stream = NULL, deployed = NULL, device = NULL ){
      self$workers <- workers
      super$initialize( stream, deployed, device )
    }
  ),

  private = list(
    .workers = NULL,

    .deploy.L1 = function(){
      super$.deploy(
        expression(
          list( workers  = .Call( "cuR_stream_queue_create", private$.workers, FALSE ) )
        )
      )
    },

    .deploy.L3 = function(){
      super$.deploy(
        expression(
          list( workers  = .Call( "cuR_stream_queue_create", private$.workers, TRUE ) )
        )
      )
    },

    .destroy = function(){
      super$.destroy(
        expression(
          .Call( "cuR_stream_queue_destroy", private$.ptrs$workers )
        )
      )
    }
  ),

  active = list(
    workers = function( val ){
      if( missing( val ) ){
        private$.workers
      }else{
        if( !is.null( private$.ptrs ) ){
          stop( "Cannot change workers: deployed stream" )
        }

        if( is.null( val ) ){
          stop( "Invalid workers parameter" )
        }

        if( !is.integer( val ) || val < 1 ){
          stop( "Invalid workers parameter" )
        }

        private$.workers <- val
      }
    }
  )
)

# Pipe class ====
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
                           context  = NULL ){
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

      super$initialize( context )
    }
  ),

  private = list(
    .update.context = function( ... ){
      # Temporary variables: $lookups take a long time, it makes sense to save
      # accessed values if used multiple times
      src        <- private$.eps$src
      dst        <- private$.eps$dst
      src.level  <- src$level
      dst.level  <- dst$level

      src.device <- src$device
      dst.device <- dst$device

      src.perm   <- private$.eps$src.perm
      dst.perm   <- private$.eps$dst.perm

      context    <- private$.eps$context
      stream     <- private$.eps$stream

      # Multi-step pipes are no longer valid:
      # L0-L3 and back
      # L3-L3 cross device

      # Deep transfers are L3-L3 transfers
      transfer.deep <- ( src.level == 3L && dst.level == 3L )

      if( transfer.deep && src.device != dst.device ){
        stop( "Cross-device pipes are not allowed" )
      }

      if( ( src.level == 0L && dst.level == 3L ) ||
          ( src.level == 3L && dst.level == 0L ) ){
        stop( "Multi-step pipes are not allowed" )
      }

      # Now that the pipe is certainly single step, we can find a dedicated
      # device
      if( src.level == 3L ){
        device <- src.device
      }else if( dst.level == 3L ){
        device <- dst.device
      }else{
        device = -1
      }

      # Source permutation
      if( !is.null( src.perm ) ){
        src.perm.level  <- src.perm$level

        if( transfer.deep ){
          if( src.perm.level != 3L ){
            stop( "Source permutation tensor is not on the correct level" )
          }

          if( src.perm$device != device ){
            stop( "Source permutation tensor is not on the correct device" )
          }
        }else{
          if( src.perm.level == 3L ){
            stop( "Source permutation tensor is not on the correct level" )
          }
        }
      }

      # Destination permutation
      if( !is.null( dst.perm ) ){
        dst.perm.level  <- dst.perm$level

        if( transfer.deep ){
          if( dst.perm.level != 3L ){
            stop( "Destination permutation tensor is not on the correct level" )
          }

          if( dst.perm$device != device ){
            stop( "Destination permutation tensor is not on the correct device" )
          }
        }else{
          if( dst.perm.level == 3L ){
            stop( "Destination permutation tensor is not on the correct level" )
          }
        }
      }

      if( src.level != 0L || dst.level != 0L ){
        if( is.null( context ) ){
          stop( "Pipe requires an L1 or L3 context" )
        }

        if( is.null( context$level ) ){
          stop( "Pipe requires an L1 or L3 context" )
        }

        if( src.level == 3L || dst.level == 3L ){
          if( context$level != 3L ){
            stop( "Pipe requires an L3 context" )
          }

          if( context$device != device ){
            stop( "Context is not on the correct device" )
          }
        }
      }

      if( !is.null( stream ) ){
        if( !is.null( stream$level ) ){
          if( src.level == 3L || dst.level == 3L ){
            if( stream$level != 3L ){
              stop( "Pipe requires an L3 stream" )
            }

            if( stream$device != device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }
      }

      # Assignements
      private$.device           <- device
      private$.params$src.level <- src.level
      private$.params$dst.level <- dst.level

      browser()
      private$.fun              <- .transfer.ptr
    }
  )
)
