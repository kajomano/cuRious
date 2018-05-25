# This is the paretn class for every object that combines tensors and does
# something with them. An example use can be seen in pipe.R or many in cublas.R.

fusion <- R6Class(
  "cuR.fusion",
  inherit = alert.recv,
  public = list(
    initialize = function(){
      lapply( private$.eps.in, function( ep ){
        if( !is.tensor( ep ) ){
          stop( "Invalid input fusion endpoint" )
        }
        ep$listener.add( self )
      })

      lapply( private$.eps.out, function( ep ){
        if( !is.tensor( ep ) ){
          stop( "Invalid output fusion endpoint" )
        }
        ep$listener.add( self )
      })

      lapply( private$.eps.opt, function( ep ){
        if( !is.null( ep ) ){
          if( !is.tensor( ep ) && !is.context( ep ) ){
            stop( "Invalid optional fusion endpoint" )
          }
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

      for( ep in private$.eps.out ){
        ep$sever.refs()
      }

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

      private$.eps.in <- lapply( private$.eps.in, function( ep ){
        ep$listener.remove()
        NULL
      })

      private$.eps.out <- lapply( private$.eps.out, function( ep ){
        ep$listener.remove()
        NULL
      })

      private$.eps.opt <- lapply( private$.eps.opt, function( ep ){
        if( !is.null( ep ) ){
          ep$listener.remove()
        }
        NULL
      })

      invisible( TRUE )
    }
  ),

  private = list(
    ##############################################################
    #   If a must-have endpoint is both an input and an output,  #
    #   it should be listed in the outputs.                      #
    #   Optional endpoints should only be read!                  #
    ##############################################################
    .eps.in  = list(), # Must-have input endpoints
    .eps.out = list(), # Must-have output endpoints
    .eps.opt = list(), # Optional endpoints

    .changed = TRUE,

    # These fields need to be filled in the .update() function
    .device  = NULL,
    .fun     = NULL,
    .params  = list(),
    # Params should be mostly filled out by the super$.update() below:

    .update = function(){
      lapply( names( private$.eps.in ), function( ep.name ){
        param.name <- paste0( ep.name, ".ptr" )
        private$.params[[param.name]] <- private$.eps.in[[ep.name]]$ptr
      })

      lapply( names( private$.eps.out ), function( ep.name ){
        param.name <- paste0( ep.name, ".ptr" )
        private$.params[[param.name]] <- private$.eps.out[[ep.name]]$ptr
      })

      lapply( names( private$.eps.opt ), function( ep.name ){
        ep <- private$.eps.opt[[ep.name]]
        param.name <- paste0( ep.name, ".ptr" )
        if( !is.null( ep ) ){
          private$.params[[param.name]] <- ep$ptr
        }else{
          private$.params[[param.name]] <- NULL
        }
      })

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
      if( missing( val ) ) return( any( c(
        is.null( private$.eps.in ),
        is.null( private$.eps.out ) ) ) )
    }
  )
)
