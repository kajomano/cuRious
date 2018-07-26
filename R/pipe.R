# .Calls: src/transfer.cpp

# Pipes are for transferring data between tensors. Pipes do most of the
# argument sanity checks at creation, and try to do the rest only when needed at
# runtime. Overhead reduction is key for smaller tasks.

# Pipe context class ====
pipe.context <- R6Class(
  "cuR.pipe.context",
  inherit = fusion.context,
  public = list(
    initialize = function( stream = NULL, level = NULL, device = NULL, workers = 4L ){
      if( !is.numeric( workers ) || length( workers ) != 1 ){
        stop( "Invalid workers parameter" )
      }

      if( as.logical( workers %% 1 ) || workers < 1 ){
        stop( "Invalid workers parameter" )
      }

      private$.workers <- as.integer( workers )

      super$initialize( stream, level, device )
    }
  ),

  private = list(
    .workers = NULL,

    .deploy.L0 = function(){
      list( workers = NULL )
    },

    .deploy.L1 = function(){
      list( workers = .Call( "cuR_stream_queue_create", private$.workers, FALSE ) )
    },

    .deploy.L3 = function(){
      list( workers = .Call( "cuR_stream_queue_create", private$.workers, TRUE ) )
    },

    .destroy.L0 = function(){
      return()
    },

    .destroy.L1 = function(){
      .Call( "cuR_stream_queue_destroy", private$.ptrs$workers )
    },

    .destroy.L3 = function(){
      .Call( "cuR_stream_queue_destroy", private$.ptrs$workers )
    }
  ),

  active = list(
    workers = function( workers ){
      self$check.destroyed()

      if( missing( workers ) ){
        private$.workers
      }else{
        if( !is.numeric( workers ) || length( workers ) != 1 ){
          stop( "Invalid workers parameter" )
        }

        if( as.logical( workers %% 1 ) || workers < 1 ){
          stop( "Invalid workers parameter" )
        }

        if( private$.workers == workers ){
          return()
        }

        private$.workers <- as.integer( workers )

        # Create the new content while the old still exists
        .ptrs <- private$.deploy()

        # Destroy old content
        private$.destroy()

        # Update
        private$.ptrs <- .ptrs

        # Alert
        private$.alert.content()
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
    .call.L0 = function( src.tensor,
                         dst.tensor,
                         src.level,
                         dst.level,
                         type,
                         dims,
                         src.dims        = NULL,
                         dst.dims        = NULL,
                         src.perm.tensor = NULL,
                         dst.perm.tensor = NULL,
                         src.span.off,
                         dst.span.off ){

      src.perm <- NULL
      dst.perm <- NULL

      if( src.span.off != 1L || obj.dims( src.tensor )[[2]] != dims[[2]] ){
        src.perm <- src.span.off:( src.span.off + dims[[2]] - 1 )
      }

      if( dst.span.off != 1L || obj.dims( dst.tensor )[[2]] != dims[[2]] ){
        dst.perm <- dst.span.off:( dst.span.off + dims[[2]] - 1 )
      }

      if( !is.null( src.perm.tensor ) ){
        if( !is.null( src.perm ) ){
          src.perm <- src.perm.tensor[ src.perm ]
        }else{
          src.perm <- src.perm.tensor
        }
      }

      if( !is.null( dst.perm.tensor ) ){
        if( !is.null( dst.perm ) ){
          dst.perm <- dst.perm.tensor[ dst.perm ]
        }else{
          dst.perm <- dst.perm.tensor
        }
      }

      # Src also must be accessed by $obj, because it needs to be signalled that
      # there is a duplicate holder of it's contents
      if( is.null( dst.perm ) ){
        private$.eps.out$dst$obj <- obj.subset( private$.eps$src$obj, src.perm )
      }else{
        if( dims[[1]] == 1L ){
          private$.eps.out$dst$obj[ dst.perm ] <- obj.subset( private$.eps$src$obj, src.perm )
        }else{
          private$.eps.out$dst$obj[, dst.perm ] <- obj.subset( private$.eps$src$obj, src.perm )
        }
      }
    },

    .call.L03 = function( src.tensor,
                          dst.tensor,
                          src.level,
                          dst.level,
                          type,
                          dims,
                          src.dims        = NULL,
                          dst.dims        = NULL,
                          src.perm.tensor = NULL,
                          dst.perm.tensor = NULL,
                          src.span.off,
                          dst.span.off,
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
    },

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

      # L0-L0 contextless calls (native R) need L0 perm vectors, others dont
      native.call <- FALSE
      if( src.level == 0L && dst.level == 0L ){
        if( is.null( context ) ){
          native.call <- TRUE
        }else if( is.null( context$level ) ){
          native.call <- TRUE
        }
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
            stop( "Source permutation tensor is not on L3" )
          }

          if( src.perm$device != device ){
            stop( "Source permutation tensor is not on the correct device" )
          }
        }else if( native.call ){
          if( src.perm.level != 0L ){
            stop( "Source permutation tensor is not on L0" )
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
            stop( "Destination permutation tensor is not on L3" )
          }

          if( dst.perm$device != device ){
            stop( "Destination permutation tensor is not on the correct device" )
          }
        }else if( native.call ){
          if( dst.perm.level != 0L ){
            stop( "Destination permutation tensor is not on L0" )
          }
        }else{
          if( dst.perm.level == 3L ){
            stop( "Destination permutation tensor is not on the correct level" )
          }
        }
      }

      # Context
      if( src.level == 3L || dst.level == 3L ){
        if( !is.null( context ) ){
          if( context$level ){
            if( context$level != 3L ){
              stop( "Pipe requires an L3 context" )
            }

            if( context$device != device ){
              stop( "Pipe context is not on the correct device" )
            }
          }
        }
      }

      # Stream
      if( !is.null( stream ) ){
        if( stream$level ){
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

      # Fun
      if( native.call ){
        private$.fun   <- private$.call.L0
        private$.sever <- FALSE
      }else{
        private$.fun   <- private$.call.L03
        private$.sever <- TRUE
      }
    }
  )
)
