# This is the parent class for every object that combines tensors and does
# something with them. An example use can be seen in pipe.R or many in cublas.R.

fusion <- R6Class(
  "cuR.fusion",
  inherit = alert.recv,
  public = list(
    alert = function( ... ){
      self$check.destroyed()
      private$.context.changed <- TRUE
      private$.content.changed <- TRUE
    },

    alert.context = function( ... ){
      self$check.destroyed()
      private$.context.changed <- TRUE
    },

    alert.content = function( ... ){
      self$check.destroyed()
      private$.content.changed <- TRUE
    },

    run = function(){
      self$check.destroyed()

      for( ep in private$.eps.out ){
        ep$sever.refs()
      }

      if( private$.context.changed ){
        private$.update.context()
      }

      if( private$.content.changed ){
        private$.update.content()
      }

      .cuda.device.set( private$.device )
      res <- do.call( private$.fun, private$.params )

      invisible( res )
    },

    destroy = function(){
      self$check.destroyed()
      private$.listener.remove <- TRUE

      for( ep in private$.eps ){
        ep$listener.remove()
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

    .context.changed = TRUE,
    .content.changed = TRUE,

    # These fields need to be filled in the .update() function
    .device  = NULL,
    .fun     = NULL,
    .params  = list(),

    .add.ep  = function( ep, ep.name, output = FALSE ){
      if( !is.null( ep ) ){
        if( !is.tensor( ep ) && !is.context( ep ) ){
          stop( "Invalid fusion endpoint" )
        }

        ep$listener.add( self, ep.name )

        if( is.null( private$.eps ) ){
          private$.eps <- list()
        }

        private$.eps[[ep.name]] <- ep

        if( output ){
          if( is.null( private$.eps.out ) ){
            private$.eps.out <- list()
          }

          private$.eps.out[[ep.name]] <- ep
        }

        invisible( TRUE )
      }
    },

    .update.context = function(){
      private$.context.changed <- FALSE
    },

    .update.content = function(){
      private$.params[ paste0( names( private$.eps ), ".ptr" ) ] <-
        lapply( private$.eps, `[[`, "ptr" )

      private$.content.changed <- FALSE
    }
  ),

  active = list(
    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.eps ) )
    }
  )
)
