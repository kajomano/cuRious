# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors, 3D tensors might be implemented
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
        if( self$get.l > 2^32-1 ){
          # TODO ====
          # Use long int or the correct R type to remove this constraint
          stop( "Tensor is too large to be stored on the GPU" )
        }

        private$under  <- TRUE
        private$tensor <- .Call( "cuR_dive_tensor",
                                 private$tensor,
                                 length(private$dims),
                                 private$dims )

        if( is.null( private$tensor ) ) stop( "Tensor could not dive" )
      }

      invisible( NULL )
    },

    surface = function(){
      if( private$under ){
        private$under  <- FALSE
        private$tensor <- .Call( "cuR_surface_tensor",
                                 private$tensor,
                                 length(private$dims),
                                 private$dims )

        if( is.null( private$tensor ) ) stop( "Tensor could not surface" )
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
        ret <- .Call( "cuR_push_tensor",
                      private$tensor,
                      obj,
                      length(private$dims),
                      private$dims )

        if( is.null( ret ) ) stop( "Tensor could not be pushed" )
      }else{
        # Here you could theoretically set an object with different dimensions,
        # but meh
        self$tensor <- obj
      }

      invisible( NULL )
    },

    pull = function(){
      if( private$under ){
        ret <- .Call( "cuR_surface_tensor",
                      private$tensor,
                      length(private$dims),
                      private$dims )

        if( is.null( ret ) ) stop( "Tensor could not be pulled" )
        ret
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

    get.l = function( val ){
      if( missing(val) ) return( prod( private$dims ) )
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
  if( is.tensor( obj ) ){
    return( obj$get.dims )
  }

  if( is.vector( obj ) ){
    return( length( obj ) )
  }else if( is.matrix( obj )){
    return( c( nrow( obj ), ncol( obj ) ) )
  }else{
    stop( "Unsupported R object for tensor conversion" )
  }
}

is.tensor <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "tensor" %in% class( obj )
  })
}

is.under <- function( ... ){
  if( !all( is.tensor( ... ) ) ){
    stop( "Not all objects are tensors" )
  }

  tenss <- list( ... )
  sapply( tenss, function( tens ){
    tens$is.under
  })
}
