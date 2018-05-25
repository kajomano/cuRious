# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# Level 0: R object (        host memory, double, int, int  )
# Level 1: C array  (        host memory, float,  int, bool )
# Level 2: C array  ( pinned host memory, float,  int, bool )
# Level 3: C array  (      device memory, float,  int, bool )

# Tensor class ====
tensor <- R6Class(
  "cuR.tensor",
  inherit = alert.send,
  public = list(
    initialize = function( data   = NULL,
                           level  = NULL,
                           dims   = NULL,
                           type   = NULL,
                           init   = c( "copy", "mimic", "wrap" ),
                           device = NULL
    ){
      init <- match.arg( init, c( "copy", "mimic", "wrap" ), T )[[1]]

      # If data is not given
      if( is.null( data ) ){
        if( is.null( dims ) ){
          private$.dims <- c( 1L, 1L )
        }else{
          private$.dims <- check.dims( dims )
        }

        if( is.null( type ) ){
          private$.type <- "n"
        }else{
          private$.type <- check.type( type )
        }

        if( is.null( level ) ){
          private$.level <- 0L
        }else{
          private$.level <- check.level( level )
        }

        if( is.null( device ) ){
          private$.device <- cuda.device.default.get()
        }else{
          private$.device <- check.device( device )
        }

      # If data is supported
      }else{
        if( !is.null( dims ) ){
          stop( "Dims can not be defined when data is supplied" )
        }

        if( !is.null( type ) ){
          stop( "Type can not be defined when data is supplied" )
        }

        if( is.obj( data ) ){
          private$.dims <- obj.dims( data )
          private$.type <- obj.type( data )

          if( is.null( level ) ){
            private$.level <- 0L
          }else{
            private$.level <- check.level( level )
          }

          if( is.null( device ) ){
            private$.device <- cuda.device.default.get()
          }else{
            private$.device <- check.device( device )
          }
        }else if( is.tensor( data ) ){
          private$.dims  <- data$dims
          private$.type  <- data$type

          if( is.null( level ) ){
            private$.level <- data$level
          }else{
            private$.level <- check.level( level )
          }

          if( is.null( device ) ){
            private$.device <- data$device
          }else{
            private$.device <- check.device( device )
          }
        }else{
          stop( "Invalid data argument" )
        }
      }

      # Initialize the tensor according to init
      if( is.null( data ) ){
        private$.ptr <- private$.create.ptr()
        self$clear()
      }else{
        switch(
          init,
          copy = {
            private$.ptr <- private$.create.ptr()

            if( is.tensor( data ) ){
              data.tensor <- data
            }else{
              data.tensor <- tensor$new( data, init = "wrap" )
            }

            transfer( data.tensor, self )
          },
          mimic = {
            private$.ptr <- private$.create.ptr()
            self$clear()
          },
          wrap = {
            if( is.obj( data ) ){
              private$.ptr <- data
            }else{
              stop( "Tensors are not wrappable" )
            }
          }
        )
      }
    },

    # These functions are only there for objs, are essentially interfaces
    # to R
    push = function( obj ){
      self$check.destroyed()

      obj <- check.obj( obj )
      private$.match.dims( obj )
      private$.match.type( obj )

      obj.tensor <- tensor$new( obj,  init = "wrap" )
      transfer( obj.tensor, self )
    },

    pull = function(){
      self$check.destroyed()

      tmp <- tensor$new( self, 0L, init = "mimic" )
      transfer( self, tmp )

      tmp$ptr
    },

    clear = function(){
      self$sever.refs()
      .Call( paste0("cuR_clear_tensor_", private$.level, "_", private$.type ),
             private$.ptr,
             private$.dims )

      invisible( TRUE )
    },

    destroy = function(){
      self$check.destroyed()
      private$.destroy.ptr()
      private$.alert()
    },

    check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The tensor is destroyed" )
      }

      invisible( TRUE )
    },

    sever.refs = function(){
      if( private$.level == 0L ){
        private$.ptr[[1]] <- private$.ptr[[1]]
      }

      invisible( TRUE )
    }
  ),

  private = list(
    .ptr    = NULL,
    .level  = NULL,
    .dims   = NULL,
    .type   = NULL,
    .device = NULL,

    .create.ptr = function( level = private$.level ){
      if( prod( private$.dims ) > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Object is too large" )
      }

      switch(
        as.character( level ),
        `0` = obj.create( private$.dims, private$.type ),
        `1` = .Call( paste0("cuR_create_tensor_1_", private$.type ), private$.dims ),
        `2` = .Call( paste0("cuR_create_tensor_2_", private$.type ), private$.dims ),
        `3` = {
          .cuda.device.set( private$.device )
          .Call( paste0("cuR_create_tensor_3_", private$.type ), private$.dims )
        }
      )
    },

    .destroy.ptr = function(){
      switch(
        as.character( private$.level ),
        `1` = .Call( paste0("cuR_destroy_tensor_1_", private$.type ), private$.ptr ),
        `2` = .Call( paste0("cuR_destroy_tensor_2_", private$.type ), private$.ptr ),
        `3` = .Call( paste0("cuR_destroy_tensor_3_", private$.type ), private$.ptr )
      )

      private$.ptr <- NULL
    },

    .match.dims = function( obj ){
      if( !identical( obj.dims( obj ), private$.dims ) ) stop( "Dims do not match" )
    },

    .match.type = function( obj ){
      if( obj.type( obj ) != private$.type ) stop( "Types do not match" )
    }
  ),

  active = list(
    ptr = function( val ){
      self$check.destroyed()

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        if( private$.level != 0L ){
          stop( "Not surfaced, direct tensor access denied" )
        }

        val <- check.obj( val )
        private$.match.dims( val )
        private$.match.type( val )

        private$.ptr <- val
      }
    },

    dims = function( val ){
      self$check.destroyed()
      if( missing( val ) ) return( private$.dims )
    },

    l = function( val ){
      if( missing( val ) ) return( as.integer( prod( self$dims ) ) )
    },

    type = function( val ){
      self$check.destroyed()
      if( missing( val ) ) return( private$.type )
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

        # Create a placeholder and copy
        tmp <- tensor$new( self, level, init = "mimic" )
        transfer( self, tmp )

        # Free old memory
        # This command also alert()s
        self$destroy()

        # Update
        private$.ptr   <- tmp$ptr
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

        if( private$.level == 3L ){
          # Create a placeholder and copy
          tmp <- tensor$new( self, 3L, init = "mimic", device = device )
          transfer( self, tmp )

          # Free old memory
          # This command also alert()s
          self$destroy()

          # Update
          private$.ptr <- tmp$ptr
        }else{
          private$.alert()
        }

        private$.device <- device
      }
    },

    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.ptr ) )
    },

    is.read.only = function( val ){
      self$check.destroyed()
      if( missing( val ) ) return( private$.read.o )
    }
  )
)
