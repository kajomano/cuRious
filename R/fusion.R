# fusion.context ====
fusion.context <- R6Class(
  "cuR.fusion.context",
  inherit = .alert.send.recv,
  public = list(
    initialize = function( stream = NULL, deployed = TRUE, device = NULL ){
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
      print( "alerted" )
      self$destroy()
      super$alert.context( name )
    }
  ),

  private = list(
    .stream = NULL,

    .deploy = function(){
      # super$.deploy( expression( list( test = 1 ) ) )
      stop( "Deploying is not implemented" )
      # Should call super$.deploy()
    },

    .destroy = function(){
      # super$.destroy( expression( NULL ) )
      stop( "Destroying is not implemented" )
      # Should call super$.destroy()
    },

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

# fusion ====
fusion <- R6Class(
  "cuR.fusion",
  inherit = .alert.recv,
  public = list(
    initialize = function( context ){
      private$.add.ep( context, "context" )
      private$.add.ep( context$stream, "stream" )
    },

    run = function(){
      # self$check.destroyed()
      #
      # for( ep in private$.eps.out ){
      #   ep$sever()
      # }
      #
      # if( length( private$.context.changed ) ){
      #   private$.update.context( private$.context.changed )
      #   private$.context.changed <- NULL
      # }
      #
      # if( length( private$.content.changed ) ){
      #   private$.update.content( private$.content.changed )
      #   private$.content.changed <- NULL
      # }
      #
      # .cuda.device.set( private$.device )
      # res <- do.call( private$.fun, private$.params )
      #
      # invisible( res )
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
      # TODO ====
      browser()

      ptrs <- unlist( lapply( private$.eps[ names ], `[[`, "ptrs" ) )
      private$.params[ paste0( names( ptrs ), ".ptr" ) ] <- ptrs
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
