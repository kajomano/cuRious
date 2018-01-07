# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors, 3D tensors might be implemented
# later for convolutional networks and/or Dropout/connect. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# TODO ====
# Check for missing values

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(
    initialize = function( obj ){
      private$dims <- get.dims( obj )

      # Store tensor, force storage type
      private$tensor <- obj
      if( storage.mode( private$tensor ) != "double" ){
        warning( "Supported object is not double precision" )
        storage.mode( private$tensor ) <- "double"
      }
    },

    dive = function(){
      if( private$under ){
        return( invisible( FALSE ) )
      }

      obj <- private$tensor
      private$tensor <- private$create.tensor()
      private$push.sync( obj )

      private$under  <- TRUE

      invisible( TRUE )
    },

    surface = function(){
      if( !private$under ){
        return( invisible( private$tensor ) )
      }

      ret <- private$pull.sync()

      private$destroy.tensor()
      self$destroy.stage()
      private$tensor <- ret
      private$under  <- FALSE

      invisible( ret )
    },

    push = function( obj ){
      private$check.dims( obj )

      # Set correct storage type
      if( storage.mode( obj ) != "double" ){
        warning( "Supported object is not double precision" )
        storage.mode( obj ) <- "double"
      }

      if( self$is.under ){
        private$push.sync( obj )
      }else{
        private$tensor <- obj
      }

      invisible( TRUE )
    },

    pull = function(){
      if( private$under ){
        ret <- private$pull.sync()
        if( is.null( ret ) ) stop( "Tensor could not be pulled" )
        ret
      }else{
        private$tensor
      }
    },

    # TODO ====
    # Do multithreading for (pre)processing on the CPU

    push.preproc = function( obj ){
      private$check.staged.under()
      private$check.dims( obj )

      # Set correct storage type
      if( storage.mode( obj ) != "double" ){
        warning( "Supported object is not double precision" )
        storage.mode( obj ) <- "double"
      }

      ret <- .Call( "cuR_push_preproc",
                    obj,
                    self$get.l,
                    private$stage )

      if( is.null( ret ) ) stop( "Tensor could not be preprocessed" )

      invisible( TRUE )
    },

    push.fetch.async = function( stream ){
      private$check.staged.under()
      check.cuda.stream.active( stream )

      ret <- .Call( "cuR_push_fetch_async",
                    private$stage,
                    self$get.l,
                    private$tensor,
                    stream$get.stream )

      if( is.null( ret ) ) stop( "Tensor could not be fetched" )

      invisible( TRUE )
    },

    pull.prefetch.async = function( stream ){
      private$check.staged.under()
      check.cuda.stream.active( stream )

      ret <- .Call( "cuR_pull_prefetch_async",
                    private$stage,
                    self$get.l,
                    private$tensor,
                    stream$get.stream )

      if( is.null( ret ) ) stop( "Tensor could not be prefetched" )

      invisible( TRUE )
    },

    pull.proc = function(){
      private$check.staged.under()

      ret <- .Call( "cuR_pull_proc",
                    self$get.dims,
                    private$stage )

      if( is.null( ret ) ) stop( "Tensor could not be processed" )

      ret
    },

    create.stage = function(){
      if( self$is.staged ){
        return( invisible( FALSE ) )
      }

      private$stage <- .Call( "cuR_create_stage", self$get.l )

      if( is.null( private$stage ) ){
        stop( "Tensor could not be staged" )
      }

      invisible( TRUE )
    },

    destroy.stage = function(){
      if( !self$is.staged ){
        return( invisible( FALSE ) )
      }

      .Call( "cuR_destroy_stage", private$stage )
      private$stage <- NULL

      invisible( TRUE )
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
    },

    push.sync = function( obj ){
      if( self$is.staged ){
        buffer <- private$stage
      }else{
        buffer <- .Call( "cuR_create_buffer", self$get.l )
        if( is.null( buffer ) ) stop( "Buffer could not be created" )
      }

      ret <- .Call( "cuR_push_preproc",
                    obj,
                    self$get.l,
                    buffer )
      if( is.null( ret ) ) stop( "Tensor could not be preprocessed" )

      ret <- .Call( "cuR_push_fetch",
                    buffer,
                    self$get.l,
                    private$tensor )
      if( is.null( ret ) ) stop( "Tensor could not be fetched" )

      if( !self$is.staged ){
        .Call( "cuR_destroy_buffer", buffer )
      }
    },

    pull.sync = function(){
      if( self$is.staged ){
        buffer <- private$stage
      }else{
        buffer <- .Call( "cuR_create_buffer", self$get.l )
        if( is.null( buffer ) ) stop( "Buffer could not be created" )
      }

      ret <- .Call( "cuR_pull_prefetch",
                    buffer,
                    self$get.l,
                    private$tensor )
      if( is.null( ret ) ) stop( "Tensor could not be prefetched" )

      ret <- .Call( "cuR_pull_proc",
                    self$get.dims,
                    buffer )
      if( is.null( ret ) ) stop( "Tensor could not be processed" )

      if( !self$is.staged ){
        .Call( "cuR_destroy_buffer", buffer )
      }

      ret
    },

    check.dims = function( obj ){
      if( !identical( private$dims, get.dims( obj ) ) ){
        stop( "Dimensions do not match" )
      }
    },

    check.staged.under = function(){
      if( !self$is.staged ){
        stop( "Tensor is not staged" )
      }

      if( !self$is.under ){
        stop( "Tensor is not under" )
      }
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
    # R vectors are functionally single column matrices, thats why
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
