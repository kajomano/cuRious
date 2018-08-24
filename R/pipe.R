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
                           src.perms = NULL,
                           dst.perms = NULL,
                           context   = NULL ){

      # Sanity and dim checks
      src.span <- tensor.span$new( src )
      dst.span <- tensor.span$new( dst )

      if( src.span$tensor$type != dst$tensor$type ){
        stop( "Tensor types do not match" )
      }

      if( length( src.perms ) > .max.array.rank ){
        stop( "Invalid src.perms argument" )
      }

      src.dims <- tensor.span$span.dims

      src.perm.spans <- lapply( 1:.max.array.rank, function( rank ){
        if( length( src.perms ) < rank ){
          return( NULL )
        }

        if( is.null( src.perms[[rank]] ) ){
          return( NULL )
        }

        if( rank > src.span$rank ){
          stop( "Out of rank src.perm" )
        }

        src.perm.span <- tensor.span$new( src.perms[[rank]] )

        if( src.perm.span$tensor$type != "i" || src.perm.span$rank != 1 ){
          stop( "Invalid src.perm" )
        }

        src.dims[[rank]] <- src.perm.span$span.dims[[1]]
        src.perm.span
      })

      dst.dims <- tensor.span$span.dims

      dst.perm.spans <- lapply( 1:.max.array.rank, function( rank ){
        if( length( dst.perms ) < rank ){
          return( NULL )
        }

        if( is.null( dst.perms[[rank]] ) ){
          return( NULL )
        }

        if( rank > src.span$rank ){
          stop( "Out of rank dst.perm" )
        }

        dst.perm.span <- tensor.span$new( dst.perms[[rank]] )

        if( dst.perm.span$tensor$type != "i" || dst.perm.span$rank != 1 ){
          stop( "Invalid dst.perm" )
        }

        dst.dims[[rank]] <- dst.perm.span$span.dims[[1]]
        dst.perm.span
      })

      if( !identical( src.dims, dst.dims ) ){
        stop( "Non-matching dimensions in pipe" )
      }

      # Assignments
      private$.add.tensor.ep( src.span, "src" )
      private$.add.tensor.ep( dst.span, "dst", TRUE )

      for( rank in 1:.max.array.rank ){
        private$.add.tensor.ep( src.perm.spans[[rank]], paste0( "src.perm.", rank ) )
        private$.add.tensor.ep( dst.perm.spans[[rank]], paste0( "dst.perm.", rank ) )
      }

      super$initialize( context )
    }
  ),

  private = list(
    .call.L0 = function( src.tensor,
                         src.wrap,
                         src.level,
                         dst.tensor,
                         dst.wrap,
                         dst.level,
                         src.perm.1.tensor    = NULL,
                         src.perm.1.wrap      = NULL,
                         src.perm.2.tensor    = NULL,
                         src.perm.2.wrap      = NULL,
                         dst.perm.1.tensor    = NULL,
                         dst.perm.1.wrap      = NULL,
                         dst.perm.2.tensor    = NULL,
                         dst.perm.2.wrap      = NULL,
                         context.workers      = NULL,
                         stream.queue         = NULL ){

      # Save original and set stored dims, set spans
      src.dims.orig         <- private$.eps$src$dims
      private$.eps$src$dims <- src.wrap[, 1]

      dst.dims.orig         <- private$.eps$dst$dims
      private$.eps$dst$dims <- dst.wrap[, 1]

      if( !is.null( src.perm.1.tensor ) ){
        src.perm.1.dims.orig         <- private$.eps$src.perm.1$dims
        private$.eps$src.perm.1$dims <- src.perm.1.wrap[, 1]
      }

      if( !is.null( src.perm.2.tensor ) ){
        src.perm.2.dims.orig         <- private$.eps$src.perm.2$dims
        private$.eps$src.perm.2$dims <- src.perm.2.wrap[, 1]
      }

      if( !is.null( dst.perm.1.tensor ) ){
        dst.perm.1.dims.orig         <- private$.eps$dst.perm.1$dims
        private$.eps$dst.perm.1$dims <- dst.perm.1.wrap[, 1]
      }

      if( !is.null( dst.perm.2.tensor ) ){
        dst.perm.2.dims.orig         <- private$.eps$dst.perm.2$dims
        private$.eps$dst.perm.2$dims <- dst.perm.2.wrap[, 1]
      }

      # Spans and perms
      src.span.1 <- NULL
      src.span.2 <- NULL
      dst.span.1 <- NULL
      dst.span.2 <- NULL

      if( src.wrap[ 1, 1 ] != src.wrap[ 1, 3 ] ){
        src.span.1 <- ( src.wrap[ 1, 2 ] + 1L ):( src.wrap[ 1, 2 ] + src.wrap[ 1, 3 ] )
      }

      if( src.wrap[ 2, 1 ] != src.wrap[ 2, 3 ] ){
        src.span.2 <- ( src.wrap[ 2, 2 ] + 1L ):( src.wrap[ 2, 2 ] + src.wrap[ 2, 3 ] )
      }

      if( dst.wrap[ 1, 1 ] != dst.wrap[ 1, 3 ] ){
        dst.span.1 <- ( dst.wrap[ 1, 2 ] + 1L ):( dst.wrap[ 1, 2 ] + dst.wrap[ 1, 3 ] )
      }

      if( dst.wrap[ 2, 1 ] != dst.wrap[ 2, 3 ] ){
        dst.span.2 <- ( dst.wrap[ 2, 2 ] + 1L ):( dst.wrap[ 2, 2 ] + dst.wrap[ 2, 3 ] )
      }

      src.perm.1 <- NULL
      src.perm.2 <- NULL
      dst.perm.1 <- NULL
      dst.perm.2 <- NULL

      if( !is.null( src.perm.1.tensor ) ){
        if( src.perm.1.wrap[ 1, 1 ] != src.perm.1.wrap[ 1, 3 ] ){
          src.perm.1 <- src.perm.1.tensor[ ( src.perm.1.wrap[ 1, 2 ] + 1L ):( src.perm.1.wrap[ 1, 2 ] + src.perm.1.wrap[ 1, 3 ] ) ]
        }else{
          src.perm.1 <- src.perm.1.tensor
        }
      }

      if( !is.null( src.perm.2.tensor ) ){
        if( src.perm.2.wrap[ 1, 1 ] != src.perm.2.wrap[ 1, 3 ] ){
          src.perm.2 <- src.perm.2.tensor[ ( src.perm.2.wrap[ 1, 2 ] + 1L ):( src.perm.2.wrap[ 1, 2 ] + src.perm.2.wrap[ 1, 3 ] ) ]
        }else{
          src.perm.2 <- src.perm.2.tensor
        }
      }

      if( !is.null( dst.perm.1.tensor ) ){
        if( dst.perm.1.wrap[ 1, 1 ] != dst.perm.1.wrap[ 1, 3 ] ){
          dst.perm.1 <- dst.perm.1.tensor[ ( dst.perm.1.wrap[ 1, 2 ] + 1L ):( dst.perm.1.wrap[ 1, 2 ] + dst.perm.1.wrap[ 1, 3 ] ) ]
        }else{
          dst.perm.1 <- dst.perm.1.tensor
        }
      }

      if( !is.null( dst.perm.2.tensor ) ){
        if( dst.perm.2.wrap[ 1, 1 ] != dst.perm.2.wrap[ 1, 3 ] ){
          dst.perm.2 <- dst.perm.2.tensor[ ( dst.perm.2.wrap[ 1, 2 ] + 1L ):( dst.perm.2.wrap[ 1, 2 ] + dst.perm.2.wrap[ 1, 3 ] ) ]
        }else{
          dst.perm.2 <- dst.perm.2.tensor
        }
      }

      # if( !is.null( src.perm.tensor ) ){
      #   if( !is.null( src.perm ) ){
      #     src.perm <- src.perm.tensor[ src.perm ]
      #   }else{
      #     src.perm <- src.perm.tensor
      #   }
      # }
      #
      # if( !is.null( dst.perm.tensor ) ){
      #   if( !is.null( dst.perm ) ){
      #     dst.perm <- dst.perm.tensor[ dst.perm ]
      #   }else{
      #     dst.perm <- dst.perm.tensor
      #   }
      # }
      #
      # # Src also must be accessed by $obj, because it needs to be signalled that
      # # there is a duplicate holder of it's contents
      # if( is.null( dst.perm ) ){
      #   private$.eps.out$dst$obj <- obj.subset( private$.eps$src$obj, src.perm )
      # }else{
      #   if( dims[[1]] == 1L ){
      #     private$.eps.out$dst$obj[ dst.perm ] <- obj.subset( private$.eps$src$obj, src.perm )
      #   }else{
      #     private$.eps.out$dst$obj[, dst.perm ] <- obj.subset( private$.eps$src$obj, src.perm )
      #   }
      # }

      # Restore original dims
      private$.eps$src$dims <- src.dims.orig
      private$.eps$dst$dims <- dst.dims.orig

      if( !is.null( src.perm.1.tensor ) ){
        private$.eps$src.perm.1$dims <- src.perm.1.dims.orig
      }

      if( !is.null( src.perm.2.tensor ) ){
        private$.eps$src.perm.2$dims <- src.perm.2.dims.orig
      }

      if( !is.null( dst.perm.1.tensor ) ){
        private$.eps$dst.perm.1$dims <- dst.perm.1.dims.orig
      }

      if( !is.null( dst.perm.2.tensor ) ){
        private$.eps$dst.perm.2$dims <- dst.perm.2.dims.orig
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

      context    <- private$.eps$context
      stream     <- private$.eps$stream

      # Transfer steps and levels
      src.level  <- src$level
      dst.level  <- dst$level

      src.device <- src$device
      dst.device <- dst$device

      # Transfer levels
      transfer.deep    <- ( src.level == 3L && dst.level == 3L )
      transfer.shallow <- ( src.level != 3L && dst.level != 3L )
      # transfer.cross   <- ( !transfer.deep  && !transfer.shallow )

      # Multi-step transfers are no longer valid:
      # L0-L3 and back
      # L3-L3 cross device
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
        }else if( context$level ){
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

      # Permutations
      perms <- private$.eps[ paste0(
        c( "src", "dst" ),
        ".perm.",
        rep( 1:.max.array.rank, each = 2 ) ) ]

      for( perm in perms ){
        if( !is.null( perm ) ){
          perm.level  <- perm$level

          if( transfer.deep ){
            if( perm.level != 3L ){
              stop( "Permutation tensor is not on L3" )
            }

            if( perm$device != device ){
              stop( "Permutation tensor is not on the correct device" )
            }
          }else if( native.call ){
            if( perm.level != 0L ){
              stop( "Permutation tensor is not on L0" )
            }
          }else if( shallow.transfer ){
            if( perm.level == 3L ){
              stop( "Permutation tensor is not in host memory" )
            }
          }else{
            stop( "Permutation is not allowed on cross device-host transfers" )
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
