# Parent class to cuRious objects that hold persistent memory pointers

.container <- R6Class(
  "cuR.container",
  public = list(
    check.destroyed = function(){
      if( is.null( private$.ptrs ) ){
        stop( "Container contents are destroyed" )
      }

      invisible( TRUE )
    }
  ),

  private = list(
    .ptrs   = NULL,
    .device = NULL,

    # Unevaluated expressions
    .deploy = function( expr ){
      .cuda.device.set( private$.device )
      private$.ptrs <- eval( expr )
    },

    .destroy = function( expr ){
      .cuda.device.set( private$.device )
      eval( expr )
      private$.ptrs <- NULL
    }
  ),

  active = list(
    ptrs = function( val ){
      if( missing( val ) ){
        return( private$.ptrs )
      }else{
        stop( "Container contents are not directly settable" )
      }
    },

    device = function( device ){
      if( missing( device ) ){
        return( private$.device )
      }else{
        if( !is.null( private$.ptrs ) ){
          stop( "Cannot change device: deployed container" )
        }

        device <- check.device( device )
        private$.device <- device
      }
    },

    is.destroyed = function( val ){
      if( missing( val ) ){
        return( is.null( private$.ptrs ) )
      }
    }
  )
)
