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

# Terminology ====
# Tensor: this whole wrapper object, but also the data stored on the GPU memory
# Obj: R vectors and matrices
# Buffer:
# Dive(): allocates space on the GPU memory and moves the obj to a tensor
# Surface(): moves the tensor back to the obj, and frees up GPU memory
# Tensor being under: if the data is on the GPU memory, the tensor is under
# Push(): replaces data in the tensor if under, or in the obj if not
# Pull(): extracts data from the tensor if under, or from the obj if not

# Tensor class ====
tensor <- R6Class(
  "tensor",
  public = list(
    threads = 4,      # Set threads to 1 to turn off multithreading

    initialize = function( obj ){
      private$dims <- get.dims( obj )
      private$obj <- private$create.dummy()

      # Force storage type
      if( storage.mode( obj ) != "double" ){
        warning( "Supported object is not double precision" )
        storage.mode( obj ) <- "double"
      }

      # Copy the data (in C), to not have soft copies that
      # could later be messed up by pull()
      .Call( "cuR_copy_obj", obj, private$obj, self$get.l )

      invisible( TRUE )
    },

    dive = function(){
      if( self$is.under ){
        return( invisible( FALSE ) )
      }

      private$create.tensor()
      private$push.sync( private$obj )

      invisible( TRUE )
    },

    surface = function(){
      if( !self$is.under ){
        return( invisible( FALSE ) )
      }

      private$pull.sync( private$obj )
      private$destroy.tensor()
      self$destroy.stage()

      invisible( TRUE )
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
        .Call( "cuR_copy_obj", obj, private$obj, self$get.l )
      }

      invisible( TRUE )
    },

    pull = function( obj = NULL ){
      if( is.null(obj) ){
        obj <- private$create.dummy()
      }else{
        private$check.dims( obj )
      }

      if( self$is.under ){
        private$pull.sync( obj )
      }else{
        .Call( "cuR_copy_obj", private$obj, obj, self$get.l )
      }

      obj
    },

    push.preproc = function( obj ){
      private$check.staged.under()
      private$check.dims( obj )

      # Set correct storage type
      if( storage.mode( obj ) != "double" ){
        warning( "Supported object is not double precision" )
        storage.mode( obj ) <- "double"
      }

      .Call( "cuR_push_preproc",
             obj,
             self$get.l,
             private$stage,
             self$threads )

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

    pull.proc = function( obj = NULL ){
      private$check.staged.under()

      if( is.null(obj) ){
        obj <- private$create.dummy()
      }else{
        private$check.dims( obj )
      }

      .Call( "cuR_pull_proc",
             obj,
             self$get.l,
             private$stage,
             self$threads )

      obj
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
    obj   = NULL,
    dims   = NULL,
    stage  = NULL,

    create.tensor = function(){
      if( self$get.l > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Tensor is too large to be stored on the GPU" )
      }

      private$tensor <- .Call( "cuR_create_tensor", self$get.l )
      if( is.null( private$tensor ) ) stop( "Tensor could not be created" )
    },

    destroy.tensor = function(){
      .Call( "cuR_destroy_tensor", private$tensor )
      private$tensor <- NULL
    },

    create.dummy = function(){
      create.dummy( private$dims )
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
                    buffer,
                    self$threads )
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

    pull.sync = function( obj ){
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
                    obj,
                    self$get.l,
                    buffer,
                    self$threads )
      if( is.null( ret ) ) stop( "Tensor could not be processed" )

      if( !self$is.staged ){
        .Call( "cuR_destroy_buffer", buffer )
      }
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
      if( missing(val) ) return( !is.null( private$tensor ) )
    },

    is.staged = function( val ){
      if( missing(val) ) return( !is.null( private$stage ) )
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
