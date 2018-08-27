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
      src.ranged <- tensor.ranged$new( src, rank = .max.array.rank )
      dst.ranged <- tensor.ranged$new( dst, rank = .max.array.rank )

      if( src.ranged$tensor$type != dst$tensor$type ){
        stop( "Tensor types do not match" )
      }

      if( length( src.perms ) > .max.array.rank ){
        stop( "Overlong src.perms argument" )
      }

      src.dims <- src.ranged$spans

      src.perm.ranges <- lapply( 1:.max.array.rank, function( r ){
        if( length( src.perms ) < r ){
          return( NULL )
        }

        if( is.null( src.perms[[r]] ) ){
          return( NULL )
        }

        src.perm.ranged <- tensor.ranged$new( src.perms[[r]], rank = 1L )

        if( src.perm.ranged$tensor$type != "i" ){
          stop( "Invalid src.perm type" )
        }

        src.dims[[r]] <- src.perm.ranged$spans
        src.perm.ranged
      })

      if( length( dst.perms ) > .max.array.rank ){
        stop( "Overlong dst.perms argument" )
      }

      dst.dims <- dst.ranged$spans

      dst.perm.ranges <- lapply( 1:.max.array.rank, function( r ){
        if( length( dst.perms ) < r ){
          return( NULL )
        }

        if( is.null( dst.perms[[r]] ) ){
          return( NULL )
        }

        dst.perm.ranged <- tensor.ranged$new( dst.perms[[r]], rank = 1L )

        if( dst.perm.ranged$tensor$type != "i" ){
          stop( "Invalid dst.perm type" )
        }

        dst.dims[[r]] <- dst.perm.ranged$spans
        dst.perm.ranged
      })

      if( !identical( dst.dims, dst.dims ) ){
        stop( "Non-matching dimensions in pipe" )
      }

      # Assignments
      private$.ep.tensor.add( src.ranged, "src" )
      private$.ep.tensor.add( dst.ranged, "dst", TRUE )

      for( r in 1:.max.array.rank ){
        private$.ep.tensor.add( src.perm.ranges[[r]], paste0( "src.perm.", r ) )
        private$.ep.tensor.add( dst.perm.ranges[[r]], paste0( "dst.perm.", r ) )
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
                         src.perm.1.tensor = NULL,
                         src.perm.1.wrap   = NULL,
                         src.perm.2.tensor = NULL,
                         src.perm.2.wrap   = NULL,
                         dst.perm.1.tensor = NULL,
                         dst.perm.1.wrap   = NULL,
                         dst.perm.2.tensor = NULL,
                         dst.perm.2.wrap   = NULL,
                         context.workers   = NULL,
                         stream.queue      = NULL ){

      # Save original and set stored dims, set spans
      src.dims.orig         <- private$.eps$src$dims
      private$.eps$src$dims <- src.wrap[, 1]

      # dst.dims.orig         <- private$.eps$dst$dims
      # private$.eps$dst$dims <- dst.wrap[, 1]

      # if( !is.null( src.perm.1.tensor ) ){
      #   src.perm.1.dims.orig         <- private$.eps$src.perm.1$dims
      #   private$.eps$src.perm.1$dims <- src.perm.1.wrap[, 1]
      # }
      #
      # if( !is.null( src.perm.2.tensor ) ){
      #   src.perm.2.dims.orig         <- private$.eps$src.perm.2$dims
      #   private$.eps$src.perm.2$dims <- src.perm.2.wrap[, 1]
      # }
      #
      # if( !is.null( dst.perm.1.tensor ) ){
      #   dst.perm.1.dims.orig         <- private$.eps$dst.perm.1$dims
      #   private$.eps$dst.perm.1$dims <- dst.perm.1.wrap[, 1]
      # }
      #
      # if( !is.null( dst.perm.2.tensor ) ){
      #   dst.perm.2.dims.orig         <- private$.eps$dst.perm.2$dims
      #   private$.eps$dst.perm.2$dims <- dst.perm.2.wrap[, 1]
      # }

      # # Spans
      # src.span <- .sub( src.wrap )
      # dst.span <- .sub( dst.wrap )
      #
      # # Permutations
      # src.perm <- list( TRUE, TRUE )
      # dst.perm <- list( TRUE, TRUE )
      #
      # if( !is.null( src.perm.1.tensor ) ){
      #   src.perm[[1]] <- .sub( src.perm.1.wrap, src.perm.1.tensor )
      # }
      #
      # if( !is.null( src.perm.2.tensor ) ){
      #   src.perm[[2]] <- .sub( src.perm.2.wrap, src.perm.2.tensor )
      # }
      #
      # if( !is.null( dst.perm.1.tensor ) ){
      #   dst.perm[[1]] <- .sub( dst.perm.1.wrap, dst.perm.1.tensor )
      # }
      #
      # if( !is.null( dst.perm.2.tensor ) ){
      #   dst.perm[[2]] <- .sub( dst.perm.2.wrap, dst.perm.2.tensor )
      # }
      #
      # # Shortcuts for no permutations
      # if( all( is.logical( src.perm ) ) ){
      #   src.perm <- list()
      # }
      #
      # if( all( is.logical( dst.perm ) ) ){
      #   dst.perm <- list()
      # }

      # ITT ====
      # stop( "ITT" )
      # browser()

      # Call
      # private$.eps$dst$obj <-
      #   .sub.obj(
      #     dst.span,
      #     dst.tensor,
      #     .sub.obj(
      #       dst.perm,
      #       .sub.obj(
      #         dst.span,
      #         dst.tensor
      #       ),
      #       .sub.obj(
      #         src.perm,
      #         .sub.obj(
      #           src.span,
      #           src.tensor
      #         )
      #       )
      #     )
      #   )


      # private$.eps$dst$obj <-
      #   do.call( `[<-`, c( list( dst.tensor ),
      #                      dst.span,
      #                      do.call( `[<-`, c( list( do.call( `[`, c( list( dst.tensor ), dst.span ) ) ),
      #                                         dst.perm,
      #                                         ) ) ) )
      #
      #
      #            list( test, 2:3,   do.call( `[<-`, list(    do.call( `[`, list( test, 2:3 ) )   , 1, 0 ) )   ) )

      # test[ 2:3 ][ 1 ] <- 0
      # do.call( `[<-`, list( test, 2:3,   do.call( `[<-`, list(    do.call( `[`, list( test, 2:3 ) )   , 1, 0 ) )   ) )

      # Restore original dims
      private$.eps$src$dims <- src.dims.orig
      # private$.eps$dst$dims <- dst.dims.orig

      # if( !is.null( src.perm.1.tensor ) ){
      #   private$.eps$src.perm.1$dims <- src.perm.1.dims.orig
      # }
      #
      # if( !is.null( src.perm.2.tensor ) ){
      #   private$.eps$src.perm.2$dims <- src.perm.2.dims.orig
      # }
      #
      # if( !is.null( dst.perm.1.tensor ) ){
      #   private$.eps$dst.perm.1$dims <- dst.perm.1.dims.orig
      # }
      #
      # if( !is.null( dst.perm.2.tensor ) ){
      #   private$.eps$dst.perm.2$dims <- dst.perm.2.dims.orig
      # }
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
