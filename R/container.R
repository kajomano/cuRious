# Parent class to cuRious objects that hold persistent memory pointers

.container <- R6Class(
  "cuR.container",
  public = list(
    initialize = function( level = NULL, device = NULL ){
      if( is.null( device ) ){
        private$.device <- cuda.device.default.get()
      }else{
        private$.device <- check.device( device )
      }

      if( is.null( level ) ){
        private$.level <- 0L
      }else{
        private$.level <- check.level( level )
      }

      private$.ptrs <- private$.deploy()
    },

    destroy = function(){
      private$.destroy()
      invisible( TRUE )
    },

    check.destroyed = function(){
      if( is.null( private$.ptrs ) ){
        stop( "Container contents are destroyed" )
      }

      invisible( TRUE )
    },

    is.destroyed = function(){
      return( is.null( private$.ptrs ) )
    }
  ),

  private = list(
    .ptrs   = NULL,
    .device = NULL,
    .level  = NULL,

    .deploy = function( level = private$.level ){
      if( level == 3L ){
        .cuda.device.set( private$.device )
      }

      switch(
        level + 1L,
        private$.deploy.L0(),
        private$.deploy.L1(),
        private$.deploy.L2(),
        private$.deploy.L3()
      )
    },

    .deploy.L0 = function(){
      stop( "Deployment not implemented for L0" )
    },

    .deploy.L1 = function(){
      stop( "Deployment not implemented for L1" )
    },

    .deploy.L2 = function(){
      stop( "Deployment not implemented for L2" )
    },

    .deploy.L3 = function(){
      stop( "Deployment not implemented for L3" )
    },

    .destroy = function(){
      if( is.null( private$.ptrs ) ){
        return()
      }

      if( private$.level == 3L ){
        .cuda.device.set( private$.device )
      }

      switch(
        private$.level + 1L,
        private$.destroy.L0(),
        private$.destroy.L1(),
        private$.destroy.L2(),
        private$.destroy.L3()
      )

      private$.ptrs  <- NULL
    },

    .destroy.L0 = function(){
      stop( "Destruction not implemented for L0" )
    },

    .destroy.L1 = function(){
      stop( "Destruction not implemented for L1" )
    },

    .destroy.L2 = function(){
      stop( "Destruction not implemented for L2" )
    },

    .destroy.L3 = function(){
      stop( "Destruction not implemented for L3" )
    }
  ),

  active = list(
    ptrs = function( val ){
      self$check.destroyed()

      if( missing( val ) ){
        return( private$.ptrs )
      }else{
        stop( "Container contents are not directly settable" )
      }
    },

    level = function( level ){
      self$check.destroyed()

      if( missing( level ) ){
        return( private$.level )
      }else{
        level <- check.level( level )

        if( private$.level == level ){
          return()
        }

        # Create the new content while the old still exists
        .ptrs <- private$.deploy( level = level )

        # Destroy old content
        private$.destroy()

        # Update
        private$.ptrs  <- .ptrs
        private$.level <- level
      }
    },

    device = function( device ){
      self$check.destroyed()

      if( missing( device) ){
        return( private$.device )
      }else{
        device <- check.device( device )

        if( private$.device == device ){
          return()
        }

        # Pre-setting the device instead of giving it as an argument works
        # because removing something from a device (destroy) does no longer
        # need the correct device set. If this changes, move the device
        # into an argument as in level
        private$.device <- device

        if( private$.level == 3L ){
          # Create the new content while the old still exists
          .ptrs <- private$.deploy()

          # Destroy old content
          private$.destroy()

          # Update
          private$.ptrs <- .ptrs
        }
      }
    }
  )
)
