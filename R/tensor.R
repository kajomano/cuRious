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
    },

    dive = function(){
      if( private$under ){
        stop( "Tensor is already under" )
      }

      if( self$get.l > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Tensor is too large to be stored on the GPU" )
      }

      private$tensor <- .Call( "cuR_push_tensor",
                               NULL,
                               private$tensor,
                               private$dims,
                               private$stage )

      if( is.null( private$tensor ) ) stop( "Tensor could not dive" )

      private$under  <- TRUE

      invisible( NULL )
    },

    surface = function(){
      if( !private$under ){
        stop( "Tensor is not under" )
      }

      private$tensor <- .Call( "cuR_pull_tensor",
                               private$tensor,
                               private$dims,
                               TRUE,
                               private$stage )

      if( is.null( private$tensor ) ) stop( "Tensor could not surface" )

      private$under  <- FALSE
      self$destroy.stage()

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
                      private$dims,
                      private$stage )

        if( is.null( ret ) ) stop( "Tensor could not be pushed" )
      }else{
        # Here you could theoretically push an object with different dimensions,
        # but meh
        self$tensor <- obj
      }

      invisible( NULL )
    },

    pull = function(){
      if( private$under ){
        ret <- .Call( "cuR_pull_tensor",
                      private$tensor,
                      private$dims,
                      FALSE,
                      private$stage )

        if( is.null( ret ) ) stop( "Tensor could not be pulled" )
        ret
      }else{
        self$obj
      }
    },

    create.stage = function(){
      if( self$is.staged ){
        stop( "Tensor is already staged" )
      }

      private$stage <- .Call( "cuR_create_stage", private$dims )

      if( is.null( private$stage ) ){
        stop( "Tensor could not be staged" )
      }

      invisible( NULL )
    },

    destroy.stage = function(){
      if( !self$is.staged ){
        stop( "Tensor is not staged" )
      }

      .Call( "cuR_destroy_stage", private$stage )
      private$stage <- NULL
    }
  ),

  private = list(
    tensor = NULL,
    dims   = NULL,
    stage  = NULL,
    under  = FALSE
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
    },

    is.staged = function( val ){
      if( missing(val) ) return( !is.null( private$stage ) )
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
    # R vectors are functionally single column many row matrices, thats why
    return( c( length( obj ), 1L ) )
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
