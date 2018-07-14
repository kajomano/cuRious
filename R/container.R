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
    .level  = NULL,

    # Unevaluated expressions
    .deploy.L1 = function( expr ){
      if( !is.null( private$.ptrs ) ){
        if( private$.level != 1L ){
          stop( "Container is deployed to a different level" )
        }else{
          return()
        }
      }

      private$.ptrs <- eval( expr )
      private$.level <- 1L
    },

    .deploy.L3 = function( expr ){
      if( !is.null( private$.ptrs ) ){
        if( private$.level != 3L ){
          stop( "Container is deployed to a different level" )
        }else{
          return()
        }
      }

      .cuda.device.set( private$.device )
      private$.ptrs <- eval( expr )
      private$.level <- 3L
    },

    .destroy = function( expr ){
      if( is.null( private$.ptrs ) ){
        return()
      }

      .cuda.device.set( private$.device )
      eval( expr )
      private$.ptrs  <- NULL
      private$.level <- NULL
    }
  ),

  active = list(
    ptrs = function( val ){
      if( missing( val ) ){
        if( is.null( private$.ptrs ) ){
          warning( "Container accessed without content" )
        }

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

    level = function( level ){
      if( missing( level ) ){
        return( private$.level )
      }else{
        stop( "Container level is not directly settable" )
      }
    },

    is.destroyed = function( val ){
      if( missing( val ) ){
        return( is.null( private$.ptrs ) )
      }
    }
  )
)
