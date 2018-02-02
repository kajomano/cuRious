# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# TODO ====
# Check for missing values

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(

    initialize = function( obj ){
      # Dims and type are fixed from this point on, no way to modify them
      private$dims  <- get.dims( obj )
      private$type  <- get.type( obj )
      private$level <- get.level( obj )
      private$obj   <- private$create.obj( self$get.level )

      # Copy the data (in C) even if it is an R object to not have soft copies
      # that could later be messed up by pull() or other transfers
      transfer( obj, private$obj )
    },

    transform = function( level = 0 ){
      if( self$get.level != level ){
        # No need to check anything
        # Create a placeholder and copy
        temp <- private$create.obj( level )
        transfer( private$obj, temp )

        # Forcibly destroy the previous obj and overwrite
        destroy.obj( private$obj )
        private$obj <- temp
      }

      invisible( self )
    },

    dive = function(){
      self$transform( 3 )
    },

    surface = function(){
      self$transform()
    },

    push = function( obj ){
      transfer( obj, self )
    },

    pull = function( obj = NULL ){
      transfer( self, obj )
    }
  ),

  private = list(
    dims  = NULL,
    type  = NULL,
    level = NULL,
    obj   = NULL,

    create.obj = function( level = 0 ){
      create.obj( self$get.dims, level, obj.types[[self$get.type]] )
    }
  ),

  active = list(
    get.obj = function( val ){
      if( missing(val) ) return( private$obj )
    },

    get.dims = function( val ){
      if( missing(val) ) return( private$dims )
    },

    get.type = function( val ){
      if( missing(val) ) return( private$type )
    },

    get.level = function( val ){
      if( missing(val) ) return( private$level )
    },

    get.l = function( val ){
      if( missing(val) ) return( prod( self$get.dims ) )
    },

    is.under = function( val ){
      if( missing(val) ) return( self$get.level == 3 )
    }
  )
)

# Helper functions ====
is.tensor <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "tensor" %in% class( obj )
  })
}

check.tensor <- function( ... ){
  if( !all( is.tensor( ... ) ) ){
    stop( "Not all objects are tensors" )
  }
}

is.under <- function( ... ){
  check.tensor( ... )

  tenss <- list( ... )
  sapply( tenss, function( tens ){
    tens$is.under
  })
}

check.tensor.under <- function( ... ){
  if( !all( is.under( ... ) ) ){
    stop( "Not all tensors are under" )
  }
}
