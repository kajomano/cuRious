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

      tensor <- private$create.tensor()
      ret <- .Call( "cuR_push_tensor",
                    tensor,
                    private$tensor,
                    self$get.l,
                    private$stage,
                    NULL )

      if( is.null( ret ) ) stop( "Tensor could not dive" )

      private$tensor <- tensor
      private$under  <- TRUE

      invisible( NULL )
    },

    surface = function(){
      if( !private$under ){
        stop( "Tensor is not under" )
      }

      ret <- .Call( "cuR_pull_tensor",
                    private$tensor,
                    self$get.dims,
                    private$stage,
                    NULL )

      if( is.null( ret ) ) stop( "Tensor could not surface" )

      private$destroy.tensor()
      self$destroy.stage()
      private$tensor <- ret
      private$under  <- FALSE

      invisible( NULL )
    },

    push = function( obj, stream = NULL ){
      if( !identical( private$dims, get.dims( obj ) ) ){
        stop( "Dimensions do not match" )
      }

      if( !is.null( stream ) ){
        if( !is.cuda.stream.created( stream ) ){
          stop( "The CUDA stream is not created" )
        }

        stream <- stream$get.stream
      }

      # Set correct storage type
      if( storage.mode( obj ) != "double" ){
        storage.mode( obj ) <- "double"
      }

      if( private$under ){
        ret <- .Call( "cuR_push_tensor",
                      private$tensor,
                      obj,
                      self$get.l,
                      private$stage,
                      stream )

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
                      self$get.dims,
                      private$stage,
                      NULL )

        if( is.null( ret ) ) stop( "Tensor could not be pulled" )
        ret
      }else{
        self$obj
      }
    },

    create.stage = function(){
      if( self$is.staged ){
        return( invisible( NULL ) )
      }

      private$stage <- .Call( "cuR_create_stage", private$dims )

      if( is.null( private$stage ) ){
        stop( "Tensor could not be staged" )
      }

      invisible( NULL )
    },

    destroy.stage = function(){
      if( !self$is.staged ){
        return( invisible( NULL ) )
      }

      .Call( "cuR_destroy_stage", private$stage )
      private$stage <- NULL
    }
  ),

  private = list(
    tensor = NULL,
    dims   = NULL,
    stage  = NULL,
    under  = FALSE,

    create.tensor = function(){
      if( self$get.l > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Tensor is too large to be stored on the GPU" )
      }

      ret <- .Call( "cuR_create_tensor", self$get.l )
      if( is.null( ret ) ) stop( "Tensor could not be created" )
      ret
    },

    destroy.tensor = function(){
      .Call( "cuR_destroy_tensor", private$tensor )
    }
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

check.tensor.under <- function( ... ){
  if( !all( is.under( ... ) ) ){
    stop( "Not all tensors are under" )
  }
}
