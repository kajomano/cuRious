# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors, 3D tensors might be implemented
# later for convolutional networks and/or Dropout/connect. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# TODO ====
# Check for missing values

# TODO ====
# Create CPU memory temporary storage to save data (without converting back to
# double)

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(

    initialize = function( obj ){
      private$dims   <- get.dims( obj )
      private$level  <- get.level( obj )
      private$tensor <- private$create.dummy( private$level )

      # Force storage type
      force.double( obj )

      # Copy the data (in C) even if it is an R object to not have soft copies
      # that could later be messed up by pull() or other transfers
      transfer( obj, self )

      invisible( TRUE )
    },

    transform = function( level = 0 ){
      if( private$level != level ){
        # Create placeholder
        temp.tensor <- private$create.dummy( level )

        # Call a low-level transfer, we know all arguments are in correct form
        trnsfr.ptr( private$tensor, temp.tensor )

        # Replace current tensor
        private$tensor <- temp.tensor
      }

      invisible( TRUE )
    },

    dive = function(){
      self$transform( 3 )
      invisible( TRUE )
    },

    surface = function(){
      self$transform()
      invisible( TRUE )
    },

    push = function( obj ){
      transfer( obj, self )
      invisible( TRUE )
    },

    pull = function( obj = NULL ){
      res <- transfer( self, obj )

      if( is.null(obj) ){
        return( res )
      }

      invisible( TRUE )
    }
  ),

  private = list(
    tensor = NULL,
    level  = NULL,
    dims   = NULL,

    create.dummy = function( level = 0 ){
      create.dummy( private$dims, level )
    }
  ),

  active = list(
    get.tensor = function( val ){
      if( missing(val) ) return( private$tensor )
    },

    get.dims = function( val ){
      if( missing(val) ) return( private$dims )
    },

    get.level = function( val ){
      if( missing(val) ) return( private$level )
    },

    get.l = function( val ){
      if( missing(val) ) return( prod( private$dims ) )
    },

    is.under = function( val ){
      if( missing(val) ) return( private$level == 3 )
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
