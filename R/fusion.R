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
    initialize = function( stream = NULL, deployed = TRUE, device = NULL ){
      if( !is.null( stream ) ){
        check.cuda.stream( stream )

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

      if( deployed ){
        self$deploy()
      }
    },

    deploy = function(){
      if( is.null( private$.ptrs ) ){
        if( length( private$.context.changed ) ){
          if( private$.device != private$.stream$device ){
            stop( "Not matching device with stream device" )
          }

          private$.context.changed <- NULL
        }

        private$.deploy()
      }

      invisible( TRUE )
    },

    destroy = function(){
      if( !is.null( private$.ptrs ) ){
        private$.destroy()
      }

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
          check.cuda.stream( val )
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
          !all( sapply( tensors, `[[`, "level" ) == 3L ) ){
        stop( "Not all tensors are on L0 or L3" )
      }

      under  <- ( tensors[[1]]$level == 3L )
      device <- tensors[[1]]$device

      if( under ){
        if( !all( sapply( tensors, `[[`, "device" ) == device ) ){
          stop( "Not all tensors are on the same device" )
        }
      }

      if( under ){
        context <- private$.eps$context

        if( is.null( context ) ){
          stop( "Subroutine requires an active context" )
        }else{
          if( context$is.destroyed ){
            stop( "Subroutine requires an active context" )
          }

          if( context$device != device ){
            stop( "Context is not on the correct context" )
          }
        }
      }

      stream <- private$.eps$stream

      if( !is.null( stream ) ){
        if( !stream$is.destroyed ){
          if( !under ){
            stop( "An active stream is given to a synchronous subroutine" )
          }
        }
      }

      private$.device <- device

      if( under ){
        private$.fun <- private$.L3.call
      }else{
        private$.fun <- private$.L0.call
      }
    }
  )
)
