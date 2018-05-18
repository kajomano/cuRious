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
  "tensor",
  inherit = alert.send,
  public = list(
    initialize = function( data  = NULL,
                           level = NULL,
                           dims  = c( 1L, 1L ),
                           type  = "n",
                           init  = c( "copy", "mimic", "wrap" )
    ){
      init <- match.arg( init, c( "copy", "mimic", "wrap" ) )

      # If init is not given
      if( is.null( data ) ){
        private$.dims  <- check.dims( dims )
        private$.type  <- check.type( type )

        if( is.null( level ) ){
          private$.level <- 0L
        }else{
          private$.level <- check.level( level )
        }
        # If supported
      }else{
        if( is.obj( data ) ){
          private$.dims  <- obj.dims( data )
          private$.type  <- obj.type( data )

          if( is.null( level ) ){
            private$.level <- 0L
          }else{
            private$.level <- check.level( level )
          }
        }else if( is.tensor( data ) ){
          private$.dims  <- data$dims
          private$.type  <- data$type

          if( is.null( level ) ){
            private$.level <- data$level
          }else{
            private$.level <- check.level( level )
          }
        }else{
          stop( "Invalid data argument" )
        }
      }

      # Initialize the tensor according to init
      if( is.null( data ) ){
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

    transform = function( level = 0L ){
      private$.check.destroyed()
      level <- check.level( level )

      if( private$.level != level ){
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

      invisible( self )
    },

    dive = function(){
      self$transform( 3L )
    },

    surface = function(){
      self$transform()
    },

    # These functions are only there for objs, are essentially interfaces
    # to R
    push = function( obj ){
      private$.check.destroyed()

      obj <- check.obj( obj )
      private$.compare.dims( obj )
      private$.compare.type( obj )

      obj.tensor <- tensor$new( obj,  init = "wrap" )
      transfer( obj.tensor, self )
    },

    pull = function(){
      private$.check.destroyed()

      tmp <- tensor$new( self, 0L, init = "mimic" )
      transfer( self, tmp )

      tmp$ptr
    },

    clear = function(){
      private$.check.destroyed()
      .Call( paste0("cuR_clear_tensor_", private$.level, "_", private$.type ),
             private$.ptr,
             private$.dims )
    },

    destroy = function(){
      private$.check.destroyed()
      private$.destroy.ptr()
      private$.ptr <- NULL
      private$.alert()
    }
  ),

  private = list(
    .ptr   = NULL,
    .level = NULL,
    .dims  = NULL,
    .type  = NULL,

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
        `3` = .Call( paste0("cuR_create_tensor_3_", private$.type ), private$.dims )
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

    .compare.dims = function( obj ){
      if( !identical( obj.dims( obj ), private$.dims ) ) stop( "Dims do not match" )
    },

    .compare.type = function( obj ){
      if( obj.type( obj ) != private$.type ) stop( "Types do not match" )
    },

    .check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The tensor is destroyed" )
      }
    }
  ),

  active = list(
    ptr = function( val ){
      private$.check.destroyed()

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        if( !self$is.surfaced ) stop( "Not surfaced, direct tensor access denied" )

        val <- check.object( val )
        private$.compare.dims( val )
        private$.compare.type( val )

        private$.ptr <- val
      }
    },

    dims = function( val ){
      private$.check.destroyed()
      if( missing( val ) ) return( private$.dims )
    },

    type = function( val ){
      private$.check.destroyed()
      if( missing( val ) ) return( private$.type )
    },

    level = function( val ){
      private$.check.destroyed()
      if( missing( val ) ) return( private$.level )
    },

    l = function( val ){
      if( missing( val ) ) return( as.integer( prod( self$dims ) ) )
    },

    is.surfaced = function( val ){
      if( missing( val ) ) return( self$level == 0 )
    },

    is.under = function( val ){
      if( missing( val ) ) return( self$level %in% c( 1L, 2L ) )
    },

    is.deep = function( val ){
      if( missing( val ) ) return( self$level == 3L )
    },

    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.ptr ) )
    }
  )
)
