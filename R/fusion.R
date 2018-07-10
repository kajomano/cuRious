# fusion ====
fusion <- R6Class(
  "cuR.fusion",
  inherit = .alert.recv,
  public = list(
    initialize = function( stream ){
      private$.add.ep( stream, "stream" )
    },

    run = function(){
      self$check.destroyed()

      for( ep in private$.eps.out ){
        ep$sever()
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

    .add.ep  = function( ep, ep.name, output = FALSE ){
      if( !is.null( ep ) ){
        private$.eps[[ep.name]] <- ep
        private$.subscribe( ep, ep.name )

        if( output ){
          private$.eps.out[[ep.name]] <- ep
        }
      }
    },

    .update.context = function( names ){
      stop( "Context update not implemented" )
    },

    .update.content = function( names ){
      lapply( names, function( ep.name ){
        ptrs.names <- paste0( ep.name, ".", names( private$.eps[[ ep.name ]]$ptrs ) )
        private$.params[ ptrs.names ] <- private$.eps[[ ep.name ]]$ptrs
      })
    }
  ),

  active = list(
    is.destroyed = function( val ){
      if( missing( val ) ){
        return( is.null( private$.eps ) )
      }
    }
  )
)

# fusion.context ====
fusion.context <- R6Class(
  "cuR.fusion.context",
  inherit = .alert.send.recv,
  public = list(
    initialize = function( stream = NULL, deployed = 3, device = NULL ){
      if( !is.null( stream ) ){
        check.stream( stream )

        if( !is.null( device ) ){
          if( stream$device != device ){
            stop( "Stream device and supported device does not match" )
          }
        }

        self$device <- stream$device
        private$.attach.stream( stream )
      }else{
        if( is.null( device ) ){
          self$device <- cuda.device.default.get()
        }else{
          self$device <- device
        }
      }

      if( is.null( deployed ) ){
        return()
      }else if( deployed == 1L ){
        self$deploy.L1()
      }else if( deployed == 3L ){
        self$deploy.L3()
      }else{
        stop( "Invalid deploy target level" )
      }
    },

    deploy.L1 = function(){
      if( length( private$.context.changed ) ){
        if( private$.device != private$.stream$device ){
          stop( "Not matching device with stream device" )
        }

        private$.context.changed <- NULL
      }

      private$.deploy.L1()
      invisible( TRUE )
    },

    deploy.L3 = function(){
      if( length( private$.context.changed ) ){
        if( private$.device != private$.stream$device ){
          stop( "Not matching device with stream device" )
        }

        private$.context.changed <- NULL
      }

      private$.deploy.L3()
      invisible( TRUE )
    },

    destroy = function(){
      private$.destroy()
      invisible( TRUE )
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
        private$.stream
      }else{
        if( !is.null( private$.ptrs ) ){
          stop( "Cannot change stream: deployed context" )
        }

        if( is.null( val ) ){
          private$.attach.stream( NULL )
        }else{
          check.stream( val )
          private$.attach.stream( val )
        }
      }
    }
  )
)

# contexted fusion ====
contexted.fusion <- R6Class(
  "cuR.contexted.fusion",
  inherit = fusion,
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
    }
  ),

  private = list(
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

      if( level %in% c( 1L, 3L ) ){
        if( is.null( context ) ){
          stop( "Subroutine requires an active context" )
        }else{
          if( context$is.destroyed ){
            stop( "Subroutine requires an active context" )
          }

          if( context$device != device ){
            stop( "Context is not deployed to the correct device" )
          }

          if( context$level != level ){
            stop( "Context is not deployed to the correct level" )
          }
        }
      }

      if( !is.null( stream ) ){
        if( !stream$is.destroyed ){
          if( level == 0L ){
            warning( "An active stream is given to a synchronous subroutine" )
          }
        }
      }

      private$.device <- device

      private$.fun <- switch(
        as.character( level ),
        `0`= private$.L0.call,
        `1`= private$.L1.call,
        `3`= private$.L3.call
      )
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
  )
)
