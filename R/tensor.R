# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# Level 0: R object (        host memory, double, int, int  )
# Level 1: C array  (        host memory, float,  int, bool )
# Level 2: C array  ( pinned host memory, float,  int, bool )
# Level 3: C array  (      device memory, float,  int, bool )

# TODO ====
# Check for missing values on init, pull, push

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(
    initialize = function( obj   = NULL,
                           level = 0L,
                           dims  = NULL,
                           type  = "n" ){

      # If object is not supported
      if( is.null( obj ) ){
        private$.dims <- check.dims( dims )
        private$.type <- check.type( type )
      # If supported
      }else{
        private$.dims <- obj.dims( obj )
        private$.type <- obj.type( obj )
      }

      private$.level <- check.level( level )

      # Allocate space for the tensor
      private$.ptr <- private$create.ptr()

      # Copy the data (in C) even if it is an R object, to not have soft copies
      # that could later be messed up by pull() or other transfers
      if( !is.null( obj ) ){
        .transfer.core( obj,
                        private$.ptr,
                        0L,
                        private$.level,
                        private$.type,
                        private$.dims )
      }else{
        self$clear()
      }
    },

    transform = function( level = 0L ){
      private$check.destroyed()
      level <- check.level( level )

      if( private$.level != level ){
        # Create a placeholder and copy
        tmp <- private$create.ptr( level )
        .transfer.core( private$.ptr,
                        tmp,
                        private$.level,
                        level,
                        private$.type,
                        private$.dims )

        # Update
        private$.ptr   <- tmp
        private$.level <- level
      }

      invisible( self )
    },

    dive = function(){
      private$check.destroyed()
      self$transform( 3L )
    },

    surface = function(){
      private$check.destroyed()
      self$transform()
    },

    # These functions are only there for objs, are essentially interfaces
    # to R
    push = function( obj ){
      private$check.destroyed()

      obj <- check.obj( obj )
      private$compare.dims( obj )
      private$compare.type( obj )

      .transfer.core( obj,
                      private$.ptr,
                      0L,
                      private$.level,
                      private$.type,
                      private$.dims )
    },

    pull = function(){
      private$check.destroyed()

      tmp <- private$create.ptr( 0L )
      .transfer.core( private$.ptr,
                      tmp,
                      private$.level,
                      0L,
                      private$.type,
                      private$.dims )
      tmp
    },

    clear = function(){
      private$check.destroyed()
      .Call( paste0("cuR_clear_tensor_", private$.level, "_", private$.type ),
             private$.ptr,
             private$.dims )
    },

    destroy = function(){
      private$.ptr <- NULL
    }
  ),

  private = list(
    .ptr   = NULL,
    .level = NULL,
    .dims  = NULL,
    .type  = NULL,

    create.ptr = function( level = private$.level ){
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

    compare.dims = function( obj ){
      if( !identical( obj.dims( obj ), private$.dims ) ) stop( "Dims do not match" )
    },

    compare.type = function( obj ){
      if( obj.type( obj ) != private$.type ) stop( "Types do not match" )
    },

    check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The tensor is destroyed" )
      }
    }
  ),

  active = list(
    ptr = function( val ){
      private$check.destroyed()

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        if( !self$is.surfaced ) stop( "Not surfaced, direct access denied" )

        val <- check.object( val )
        private$compare.dims( val )
        private$compare.type( val )

        private$.ptr <- val
      }
    },

    dims = function( val ){
      private$check.destroyed()
      if( missing( val ) ) return( private$.dims )
    },

    type = function( val ){
      private$check.destroyed()
      if( missing( val ) ) return( private$.type )
    },

    level = function( val ){
      private$check.destroyed()
      if( missing( val ) ) return( private$.level )
    },

    l = function( val ){
      if( missing( val ) ) return( as.integer( prod( self$dims ) ) )
    },

    is.under = function( val ){
      if( missing( val ) ) return( self$level == 3 )
    },

    is.surfaced = function( val ){
      if( missing( val ) ) return( self$level == 0 )
    },

    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.ptr ) )
    }
  )
)
