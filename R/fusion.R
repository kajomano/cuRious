# fusion.context ====
fusion.context <- R6Class(
  "cuR.fusion.context",
  inherit = .alert.send,
  public = list(
    initialize = function( stream = NULL, level = NULL, device = NULL ){
      if( !is.null( stream ) ){
        private$.stream <- check.stream( stream )

        if( is.null( level ) ){
          level <- stream$level
        }

        if( is.null( device ) ){
          device <- stream$device
        }
      }

      super$initialize( level, device )
    }
  ),

  private = list(
    .stream = NULL
  ),

  active = list(
    stream = function( val ){
      if( missing( val ) ){
        return( private$.stream )
      }else{
        stop( "Stream is not directly settable" )
      }
    }
  )
)

# fusion ====
fusion <- R6Class(
  "cuR.fusion",
  inherit = .alert.recv,
  public = list(
    initialize = function( context ){
      if( !is.null( context ) ){
        if( sub( "^(cuR\\.[^\\.]*).*", "\\1", class( self )[[1]] ) !=
            sub( "^(cuR\\.[^\\.]*).*", "\\1", class( context )[[1]] ) ){
          stop( "Context does not match this fusion" )
        }
      }

      private$.eps$context <- context
      private$.subscribe( context, "context" )

      if( !is.null( context$stream ) ){
        private$.eps$stream <- context$stream
        private$.subscribe( context$stream, "stream" )
      }
    },

    run = function(){
      self$check.destroyed()

      if( private$.sever ){
        for( ep in private$.eps.out ){
          ep$sever()
        }
      }

      if( length( private$.context.changed ) ){
        private$.update.context( private$.context.changed )
        private$.context.changed <- NULL
      }

      if( length( private$.content.changed ) ){
        private$.update.content( private$.content.changed )
        private$.content.changed <- NULL
      }

      .cuda.device.set( private$.device )
      res <- do.call( private$.fun, private$.params )

      invisible( res )
    },

    destroy = function(){
      if( is.null( private$.eps ) ){
        return( invisible( TRUE ) )
      }

      for( ep.name in names( private$.eps ) ){
        private$.unsubscribe( private$.eps[[ep.name]], ep.name )
      }

      private$.eps     <- NULL
      private$.eps.out <- NULL

      invisible( TRUE )
    },

    check.destroyed = function(){
      if( is.null( private$.eps ) ){
        stop( "The fusion is destroyed" )
      }

      invisible( TRUE )
    },

    is.destroyed = function(){
      return( is.null( private$.eps ) )
    }
  ),

  private = list(
    # Stored container handles (endpoints)
    .eps      = NULL,
    .eps.out  = NULL,

    # TODO ====
    # Tensor dims storage

    # # Stored dimension objects
    # .tens.dims = NULL,

    # These fields need to be filled in the .update.context() function
    .fun     = NULL,
    .params  = list(),
    .device  = NULL,

    .sever   = TRUE,

    .add.tensor.ep = function( tensor.span, tensor.name, output = FALSE ){
      if( !is.null( tensor.span ) ){
        # Endpoint lists and alerts
        private$.eps[[tensor.name]] <- tensor.span$tensor
        private$.subscribe( tensor.span$tensor, tensor.name )

        if( output ){
          private$.eps.out[[tensor.name]] <- tensor.span$tensor
        }
      }

      # Params
      private$.params[[ paste0( tensor.name, ".type" ) ]]      <- tensor.span$type
      private$.params[[ paste0( tensor.name, ".dims" ) ]]      <- tensor.span$dims
      private$.params[[ paste0( tensor.name, ".span.dims" ) ]] <- tensor.span$span.dims
      private$.params[[ paste0( tensor.name, ".span.offs" ) ]] <- tensor.span$span.offs
    },

    .update.context = function( ... ){

      tensors <- sapply( private$.eps, is.tensor )
      tensors <- private$.eps[ tensors ]

      if( !all( sapply( tensors, `[[`, "level" ) == 0L ) &&
          !all( sapply( tensors, `[[`, "level" ) == 1L ) &&
          !all( sapply( tensors, `[[`, "level" ) == 3L ) ){
        stop( "Not all tensors are on L0, L1 or L3" )
      }

      level   <- tensors[[1]]$level
      device  <- tensors[[1]]$device
      context <- private$.eps$context
      stream  <- private$.eps$stream

      if( level == 3L ){
        if( !all( sapply( tensors, `[[`, "device" ) == device ) ){
          stop( "Not all tensors are on the same device" )
        }
      }

      # Context
      if( level %in% c( 1L, 3L ) ){
        if( is.null( context ) ){
          stop( "Subroutine requires an L1 or L3 context" )
        }

        if( !context$level ){
          stop( "Subroutine requires an L1 or L3 context" )
        }

        if( level == 3L ){
          if( context$level != 3L ){
            stop( "Subroutine requires an L3 context" )
          }

          if( context$device != device ){
            stop( "Subroutine context is not on the correct device" )
          }
        }
      }

      # Stream
      if( !is.null( stream ) ){
        if( stream$level ){
          if( level == 3L ){
            if( stream$level != 3L ){
              stop( "Pipe requires an L3 stream" )
            }

            if( stream$device != device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }
      }

      private$.device <- device

      private$.fun <- switch(
        level + 1L,
        private$.call.L0,
        private$.call.L1,
        private$.call.L2,
        private$.call.L3
      )

      # TODO ====
      # Test this, this was private$.sever <- as.logical( level ) before

      private$.sever <- !as.logical( level )
    },

    .update.content = function( names ){
      for( ep.name in names ){
        ptrs.names <- paste0( ep.name, ".", names( private$.eps[[ ep.name ]]$ptrs ) )
        private$.params[ ptrs.names ] <- private$.eps[[ ep.name ]]$ptrs
      }
    },

    .call.L0 = function( ... ){
      stop( "L0 call not implemented" )
    },

    .call.L1 = function( ... ){
      stop( "L1 call not implemented" )
    },

    .call.L2 = function( ... ){
      stop( "L2 call not implemented" )
    },

    .call.L3 = function( ... ){
      stop( "L3 call not implemented" )
    }
  ),

  active = list(
    context = function( val ){
      if( missing( val ) ){
        return( private$.context )
      }else{
        stop( "Context is not directly settable" )
      }
    }
  )
)
