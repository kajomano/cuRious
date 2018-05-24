context <- R6Class(
  "cuR.context",
  inherit = alert.send,
  public = list(
    initialize = function( device = cuda.device.default.get() ){
      if( !is.null( device ) ){
        private$.device <- check.device( device )
        private$.ptr    <- private$.activate()
      }else{
        private$.device <- cuda.device.default.get()
      }
    },

    activate = function(){
      if( is.null( private$.ptr ) ){
        .cuda.device.set( private$.device )
        private$.ptr <- private$.activate()
        private$.alert()
      }else{
        warning( "Context is already activated" )
      }

      invisible( self )
    },

    deactivate = function(){
      if( !is.null( private$.ptr ) ){
        private$.deactivate()
        private$.ptr <- NULL
        private$.alert()
      }else{
        warning( "Context is not yet activated" )
      }

      invisible( self )
    }
  ),

  private = list(
    .ptr    = NULL,
    .device = NULL,

    .activate = function(){
      stop( "Activation is not implemented" )
    },

    .deactivate = function(){
      stop( "Deactivation is not implemented" )
    }
  ),

  active = list(
    ptr = function( val ){
      if( missing( val ) ) return( private$.ptr )
    },

    device = function( device ){
      if( missing( device ) ){
        return( private$.device )
      }else{
        device <- check.device( device )

        if( private$.device == device ){
          return()
        }

        if( self$is.active ){
          stop( "Cannot change device: active context" )
        }

        private$.device <- device
      }
    },

    is.active = function( val ){
      if( missing( val ) ) return( !is.null( private$.ptr ) )
    }
  )
)
