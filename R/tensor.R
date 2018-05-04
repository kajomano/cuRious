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
    initialize = function( obj   = NULL,
                           dims  = get.dims( obj ),
                           type  = get.type( obj ),
                           level = get.level( obj ) ){

      # Dims and type are fixed from this point on, no way to modify them
      if( is.null(  ) )
      private$dims  <- dims
      private$type  <- type

      # Rest is changeable
      private$level <- level
      private$obj   <- private$create.obj( private$level )

      # Copy the data (in C) even if it is an R object, to not have soft copies
      # that could later be messed up by pull() or other transfers
      if( !is.null( obj ) ){
        transfer( obj, private$obj )
      }else{
        self$clear()
      }
    },

    transform = function( level = 0 ){
      private$check.destroyed()

      if( private$level != level ){
        # No need to check anything
        # Create a placeholder and copy
        temp <- private$create.obj( level )
        transfer( private$obj, temp )

        # Forcibly destroy the previous obj and overwrite
        self$destroy()
        private$obj <- temp

        private$level <- level
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
      private$check.destroyed()
      transfer( obj, self )
    },

    pull = function( obj = NULL ){
      private$check.destroyed()
      transfer( self, obj )
    },

    clear = function(){
      private$check.destroyed()
      clear.obj( private$obj )
    },

    destroy = function(){
      private$check.destroyed()
      obj <- private$obj
      destroy.obj( obj )
      private$obj <- NULL
    }
  ),

  private = list(
    dims  = NULL,
    type  = NULL,
    level = NULL,
    obj   = NULL,

    create.obj = function( level = 0 ){
      create.obj( private$dims, level, private$type )
    },

    check.destroyed = function(){
      if( is.null(private$obj) ){
        stop( "The tensor was destroyed previously" )
      }
    }
  ),

  active = list(
    get.obj = function( val ){
      private$check.destroyed()
      if( missing(val) ) return( private$obj )
    },

    get.dims = function( val ){
      private$check.destroyed()
      if( missing(val) ) return( private$dims )
    },

    get.type = function( val ){
      private$check.destroyed()
      if( missing(val) ) return( private$type )
    },

    get.level = function( val ){
      private$check.destroyed()
      if( missing(val) ) return( private$level )
    },

    get.l = function( val ){
      if( missing(val) ) return( prod( self$get.dims ) )
    },

    is.under = function( val ){
      if( missing(val) ) return( self$get.level == 3 )
    },

    is.surfaced = function( val ){
      if( missing(val) ) return( self$get.level == 0 )
    },

    is.destroyed = function( val ){
      if( missing(val) ) return( is.null(private$obj) )
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

is.surfaced <- function( ... ){
  check.tensor( ... )

  tenss <- list( ... )
  sapply( tenss, function( tens ){
    tens$is.surfaced
  })
}

check.tensor.surfaced <- function( ... ){
  if( !all( is.surfaced( ... ) ) ){
    stop( "Not all tensors are surfaced" )
  }
}
