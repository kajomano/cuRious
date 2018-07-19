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

      private$.add.ep( context, "context" )
      private$.add.ep( context$stream, "stream" )
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
      self$check.destroyed()

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
    }
  ),

  private = list(
    .eps     = NULL,
    .eps.out = NULL,

    # These fields need to be filled in the .update.context() function
    .fun     = NULL,
    .params  = list(),

    .device  = NULL,

    .sever   = TRUE,

    .add.ep  = function( ep, ep.name, output = FALSE ){
      if( !is.null( ep ) ){
        private$.eps[[ep.name]] <- ep
        private$.subscribe( ep, ep.name )

        if( output ){
          private$.eps.out[[ep.name]] <- ep
        }
      }
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

        if( is.null( context$level ) ){
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
        if( !is.null( stream$level ) ){
          if( level == 0L ){
            warning( "An active stream is given to a synchronous subroutine" )
          }

          if( level == 3L ){
            if( stream$level != 3L ){
              stop( "Pipe requires an L3 stream" )
            }

            if( stream$device != device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }else{
          # TODO ====
          # Im bothered by the warnings, but also dont want to accidentally
          # forget something
          # warning( "Stream supported but inactive" )
        }
      }

      private$.device <- device

      private$.fun <- switch(
        as.character( level ),
        `0`= private$.call.L0,
        `1`= private$.call.L1,
        `3`= private$.call.L3
      )

      private$.sever <- as.logical( level )
    },

    .update.content = function( names ){
      lapply( names, function( ep.name ){
        ptrs.names <- paste0( ep.name, ".", names( private$.eps[[ ep.name ]]$ptrs ) )
        private$.params[ ptrs.names ] <- private$.eps[[ ep.name ]]$ptrs
      })
    },

    .call.L0 = function( ... ){
      stop( "L0 call not implemented" )
    },

    .call.L1 = function( ... ){
      stop( "L1 call not implemented" )
    },

    .call.L3 = function( ... ){
      stop( "L3 call not implemented" )
    }
  ),

  active = list(
    is.destroyed = function( val ){
      if( missing( val ) ){
        return( is.null( private$.eps ) )
      }
    },

    context = function( val ){
      if( missing( val ) ){
        return( private$.context )
      }else{
        stop( "Context is not directly settable" )
      }
    }
  )
)

# fusion.context ====
fusion.context <- R6Class(
  "cuR.fusion.context",
  inherit = .alert.send.recv,
  public = list(
    initialize = function( stream = NULL, device = NULL ){
      if( !is.null( stream ) ){
        check.stream( stream )
        private$.attach.stream( stream )
      }

      if( is.null( device ) ){
        if( !is.null( stream ) ){
          self$device <- stream$device
        }else{
          self$device <- cuda.device.default.get()
        }
      }else{
        self$device <- device
      }
    },

    deploy = function( level ){
      if( is.null( level ) ){
        stop( "No deployment target level" )
      }

      if( !( level %in% c( 1L, 3L ) ) ){
        stop( "Invalid deployment target level" )
      }

      if( !self$is.destroyed ){
        if( private$.level != level ){
          stop( "Context is already deployed to a different level" )
        }else{
          return()
        }
      }

      if( length( private$.context.changed ) && level == 3L ){
        if( private$.device != private$.stream$device ){
          stop( "Not matching device with stream device" )
        }

        private$.context.changed <- NULL
      }

      if( level == 1L ){
        private$.deploy.L1()
      }else if( level == 3L ){
        private$.deploy.L3()
      }

      invisible( self )
    },

    destroy = function(){
      private$.destroy()
      invisible( self )
    },

    alert.context = function( name ){
      self$destroy()
      super$alert.context( name )
    }
  ),

  private = list(
    .stream = NULL,

    .attach.stream = function( stream ){
      if( !is.null( private$.stream ) ){
        private$.unsubscribe( private$.stream, "stream" )
      }

      if( !is.null( stream ) ){
        private$.subscribe( stream, "stream" )
      }

      private$.stream <- stream
    }
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
