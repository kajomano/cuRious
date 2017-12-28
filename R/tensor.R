# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors, 3D tensors will be implemented
# later for convolutional networks and/or Dropout/connect. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(
    initialize = function( obj ){
      private$dims <- get.dims( obj )

      if( any( is.na( obj ) ) ){
        stop( "Missing data in R object" )
      }

      # Store tensor, force storage type
      private$tensor <- obj
      storage.mode( private$tensor ) <- "double"

      # Set info
      private$under <- FALSE
    },

    dive = function(){
      if( !private$under ){
        private$under  <- TRUE
        private$tensor <- .Call( "dive_tensor",
                                 private$tensor,
                                 length(private$dims),
                                 private$dims )
      }

      invisible( NULL )
    },

    surface = function(){
      if( private$under ){
        private$under  <- FALSE
        private$tensor <- .Call( "surface_tensor",
                                 private$tensor,
                                 length(private$dims),
                                 private$dims )
      }

      invisible( NULL )
    },

    push = function( obj ){
      if( !identical( private$dims, get.dims( obj ) ) ){
        stop( "Dimensions do not match" )
      }

      # Set correct storage type
      storage.mode( obj ) <- "double"

      if( private$under ){
        .Call( "push_tensor",
               private$tensor,
               obj,
               length(private$dims),
               private$dims )
      }else{
        # Here you could theoretically set an object with different dimensions,
        # but meh
        self$obj <- obj
      }

      invisible( NULL )
    },

    pull = function(){
      if( private$under ){
        .Call( "surface_tensor",
               private$tensor,
               length(private$dims),
               private$dims )
      }else{
        self$obj
      }
    }
  ),

  private = list(
    tensor  = NULL,
    dims    = NULL,
    under   = NULL
  ),

  active = list(
    get.tensor = function( val ){
      if( missing(val) ) return( private$tensor )
    },

    get.dims = function( val ){
      if( missing(val) ) return( private$dims )
    },

    is.under = function( val ){
      if( missing(val) ) return( private$under )
    }
  )
)

# Helper functions ====

# This function checks for validity too!
# The order of dims is super important!
get.dims <- function( obj ){
  if( is.vector( obj ) ){
    return( length( obj ) )
  }else if( is.matrix( obj )){
    return( c( nrow( obj ), ncol( obj ) ) )
  }else{
    stop( "Unsupported R object for tensor conversion" )
  }
}

is.tensor <- function( obj ){
  "tensor" %in% class( obj )
}
