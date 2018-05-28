# This is the parent class for every object that combines tensors and does
# something with them. An example use can be seen in pipe.R or many in cublas.R.

fusion <- R6Class(
  "cuR.fusion",
  inherit = alert.recv,
  public = list(
    run = function(){
      self$check.destroyed()

      for( ep in private$.eps.out ){
        ep$sever()
      }

      if( !is.null( private$.context.changed ) ){
        private$.update.context()
        private$.context.changed <- NULL
      }

      if( !is.null( private$.content.changed ) ){
        private$.update.content( private$.content.changed )
        private$.content.changed <- NULL
      }

      .cuda.device.set( private$.device )
      res <- do.call( private$.fun, private$.params )

      invisible( res )
    },

    destroy = function(){
      self$check.destroyed()
      private$.listener.remove <- TRUE

      for( ep in private$.eps ){
        private$.unsubscribe( ep )
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

        private$.eps[[ep.name]] <- ep
        private$.subscribe( ep, ep.name )

        if( output ){
          private$.eps.out[[ep.name]] <- ep
        }

        invisible( TRUE )
      }
    },

    .update.context = function(){
      stop( "Context update not implemented" )
    },

    .update.content = function( names ){
      private$.params[ paste0( names( private$.eps ), ".ptr" ) ] <-
        lapply( private$.eps, `[[`, "ptr" )

      print( "content updated" )
      private$.content.changed <- NULL
    }
  ),

  active = list(
    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.eps ) )
    }
  )
)
