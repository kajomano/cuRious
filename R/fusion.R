# This is the paretn class for every object that combines tensors and does
# something with them. An example use can be seen in pipe.R or many in cublas.R.

fusion <- R6Class(
  "cuR.fusion",
  inherit = alert.recv,
  public = list(
    initialize = function(){
      lapply( private$.eps.fix, function( ep ){
        check.alerting( ep )
        ep$listener.add( self )
      })

      lapply( private$.eps.fix, function( ep ){
        if( !is.null( ep ) ){
          check.alerting( ep )
          ep$listener.add( self )
        }
      })
    },

    alert = function(){
      private$.check.destroyed()
      private$.changed <- TRUE
    },

    run = function(){
      private$.check.destroyed()

      if( private$.changed ){
        private$.update()
      }

      .cuda.device.set( private$.device )
      res <- do.call( private$.fun, private$.params )

      invisible( res )
    },

    destroy = function(){
      private$.check.destroyed()
      private$.listener.remove <- TRUE

      private$.eps.fix <- lapply( private$.eps.fix, function( ep ){
        ep$listener.remove()
        NULL
      })

      private$.eps.fix <- lapply( private$.eps.fix, function( ep ){
        if( !is.null( ep ) ){
          ep$listener.remove()
        }
        NULL
      })

      invisible( TRUE )
    }
  ),

  private = list(
    .eps.fix = list(), # Must-have endpoints
    .eps.opt = list(), # Optional endpoints

    .changed = TRUE,

    # These fields need to be filled in the .update() function
    .fun     = NULL,
    .params  = list(),

    .device  = NULL,

    .update = function(){
      private$.changed <- FALSE
    },

    .check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The fusion is destroyed" )
      }
    }
  ),

  active = list(
    is.destroyed = function( val ){
      if( missing( val ) ) return( any( is.null( private$.eps.fix ) ) )
    }
  )
)
