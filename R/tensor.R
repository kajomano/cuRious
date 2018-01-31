# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors, 3D tensors might be implemented
# later for convolutional networks and/or Dropout/connect. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# TODO ====
# Check for missing values

# TODO ====
# Store important attributes in here too, as they are unmodifiable here
# and use them if they are available, trnsfr.ptr should take these as arguments

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(

    initialize = function( obj ){
      private$dims   <- get.dims( obj )
      private$tensor <- private$create.dummy( get.level( obj ) )

      # Force storage type
      force.double( obj )

      # Copy the data (in C) even if it is an R object to not have soft copies
      # that could later be messed up by pull() or other transfers
      transfer( obj, self )
    },

    transform = function( level = 0 ){
      if( self$get.level != level ){
        # Create placeholder
        temp.tensor.ptr <- private$create.dummy( level )

        # TODO ====
        # Call proper transfer here

        # Call a low-level transfer, we know all arguments are in correct form
        transfer( private$tensor, temp.tensor )

        # Replace current tensor
        private$tensor <- temp.tensor
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
    tensor = NULL,
    dims   = NULL,
    type   = NULL,
    level  = NULL,

    create.dummy = function( level = 0 ){
      create.dummy( private$dims, level )
    }
  ),

  active = list(
    get.tensor = function( val ){
      if( missing(val) ) return( private$tensor )
    },

    get.dims = function( val ){
      if( missing(val) ) return( get.dims( private$tensor ) )
    },

    get.level = function( val ){
      if( missing(val) ) return( get.level( private$tensor ) )
    },

    get.l = function( val ){
      if( missing(val) ) return( prod( private$dims ) )
    },

    is.under = function( val ){
      if( missing(val) ) return( self$get.level == 3 )
    }
  )
)

# Helper functions ====
is.tensor.ptr <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "tensor.ptr" %in% class( obj )
  })
}

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
