# Parent class to cuRious objects that hold persistent memory pointers

.container <- R6Class(
  "cuR.container",
  public = list(
    check.destroyed = function(){
      if( is.null( private$.ptr ) ){
        stop( "Container contents are destroyed" )
      }

      invisible( TRUE )
    }
  ),

  private = list(
    .ptr    = NULL,
    .device = NULL,

    # Unevaluated expressions
    .deploy = function( ptr.expr ){
      .cuda.device.set( private$.device )
      private$.ptr <- eval( ptr.expr )
    },

    .destroy = function( ptr.expr ){
      .cuda.device.set( private$.device )
      eval( ptr.expr )
      private$.ptr <- NULL
    }
  ),

  active = list(
    ptr = function( val ){
      if( missing( val ) ){
        return( private$.ptr )
      }else{
        stop( "Container contents are not directly settable" )
      }
    },

    device = function( device ){
      device = function( device ){
        if( missing( device ) ){
          return( private$.device )
        }else{
          if( !is.null( private$.ptr ) ){
            stop( "Cannot change device: deployed container" )
          }

          device <- check.device( device )
          private$.device <- device
        }
      }
    },

    is.deployed = function( val ){
      if( missing( val ) ){
        return( !is.null( private$.ptr ) )
      }
    }
  )
)
