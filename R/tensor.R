# .Calls: src/tensor.cpp
# Tensor class ====
# cuRious currently supports 1D and 2D tensors. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# Level 0: R object (        host memory, double, int, int  )
# Level 1: C array  (        host memory, float,  int, bool )
# Level 2: C array  ( pinned host memory, float,  int, bool )
# Level 3: C array  (      device memory, float,  int, bool )

# Tensors implement reference counting if the content is accessed by $obj, but
# not if accessed by $obj.unsafe, or through $ptrs

is.tensor <- function( tensor ){
  "cuR.tensor" %in% class( tensor )
}

check.tensor <- function( tensor ){
  if( !is.tensor( tensor ) ) stop( "Not a tensor" )
  invisible( tensor )
}

# TODO ====
# Add ability to create tensors from tensor.ranged-s

# Tensor ====
tensor <- R6Class(
  "cuR.tensor",
  inherit = .alert.send,

  # public ====
  public = list(
    initialize = function( data   = NULL,
                           level  = NULL,
                           dims   = NULL,
                           type   = NULL,
                           copy   = TRUE,
                           device = NULL
    ){
      # If data is not supported
      if( is.null( data ) ){
        if( is.null( dims ) )   dims   <- c( 1L, 1L )
        if( is.null( type ) )   type   <- "n"
        if( is.null( level ) )  level  <- 0L
        if( is.null( device ) ) device <- cuda.device.default.get()

      # If data is supported as an object
      }else if( is.obj( data ) ){
        if( is.null( dims ) )   dims   <- obj.dims( data )
        if( is.null( type ) )   type   <- obj.type( data )
        if( is.null( level ) )  level  <- 0L
        if( is.null( device ) ) device <- cuda.device.default.get()

        private$.dims <- obj.dims( data )
        private$.l    <- prod( private$.dims )
        private$.type <- obj.type( data )

      # If data is supported as an object
      }else if( is.tensor( data ) ){
        if( is.null( dims ) )   dims   <- data$dims
        if( is.null( type ) )   type   <- data$type
        if( is.null( level ) )  level  <- data$level
        if( is.null( device ) ) device <- data$device

        private$.dims <- data$dims
        private$.l    <- prod( private$.dims )
        private$.type <- data$type

      }else{
        stop( "Invalid data argument on init" )
      }

      # Arg checks
      dims   <- check.dims( dims )
      type   <- check.type( type )
      level  <- check.level( level )
      device <- check.device( device )

      # Match checks
      if( !is.null( private$.dims ) ){
        if( private$.l != prod( dims ) ){
          stop( "Dims mismatch on init" )
        }
      }else{
        private$.dims <- dims
        private$.l    <- prod( private$.dims )
      }

      if( !is.null( private$.type ) ){
        if( private$.type != type ){
          stop( "Type mismatch on init" )
        }
      }else{
        private$.type <- type
      }

      private$.level  <- level
      private$.device <- device

      # Initial data
      if( is.null( data ) || !copy ){
        private$.ptrs$tensor <- private$.create.tensor()
        self$clear()
      }else{
        if( is.tensor( data ) ){
          if( data$level == 0L && private$.level == 0L ){
            private$.ptrs$tensor <- data$obj
            private$.refs <- TRUE
          }else{
            private$.ptrs$tensor <- private$.create.tensor()
            private$.push( data )
          }
        }else if( is.obj( data ) ){
          if( private$.level == 0L ){
            private$.ptrs$tensor <- data
            private$.refs <- TRUE
          }else{
            private$.ptrs$tensor <- private$.create.tensor()
            private$.push( data )
          }
        }
      }

      # Recut
      self$dims <- dims
    },

    # Push takes objects or tensors, pull returns only objects
    # Reason: if you want to pull into a tensor, you can push into it this
    # tensor. If you need a new tensor while also data being pulled into it,
    # create a new tensor with data = this tensor.
    push = function( data ){
      self$sever()

      private$.check.match( data )
      private$.push( data )

      invisible( TRUE )
    },

    pull = function(){
      self$check.destroyed()
      private$.deploy( level = 0L )$tensor
    },

    clear = function(){
      self$sever()

      .Call( "cuR_tensor_clear",
             private$.ptrs$tensor,
             private$.level,
             private$.l,
             private$.type )

      invisible( TRUE )
    },

    sever = function(){
      self$check.destroyed()

      if( private$.level == 0L ){
        if( private$.refs ){
          private$.ptrs$tensor <- .obj.duplicate( private$.ptrs$tensor )
          private$.refs <- FALSE
          private$.alert.content()
        }
      }

      invisible( TRUE )
    }
  ),

  # private ====
  private = list(
    .dims    = NULL,

    # Unchanging
    .l       = NULL,
    .type    = NULL,

    # External R references
    .refs    = FALSE,

    .create.tensor = function(){
      if( private$.l > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Object is too large" )
      }

      if( !private$.level ){
        .obj.create( private$.dims, private$.type )
      }else{
        if( private$.level == 3L ){
          .cuda.device.set( private$.device )
        }

        .Call( "cuR_tensor_create",
               private$.level,
               private$.l,
               private$.type )
      }
    },

    .check.match = function( data ){
      if( is.tensor( data ) ){
        dims <- data$dims
        type <- data$type
      }else if( is.obj( data ) ){
        dims <- obj.dims( data )
        type <- obj.type( data )
      }else{
        stop( "Invalid data" )
      }

      if( !identical( dims, private$.dims ) ){
        stop( "Dims mismatch" )
      }

      if( type != private$.type ){
        stop( "Type mismatch" )
      }
    },

    .push = function( data ){
      if( is.tensor( data ) ){
        data.tensor <- data
      }else if( is.obj( data ) ){
        data.tensor <- tensor$new( data )
      }else{
        stop( "Invalid data format" )
      }

      transfer( data.tensor, self )

      if( is.obj( data ) ){
        data.tensor$destroy()
      }
    },

    .deploy.tensor = function( level ){
      .tensor <- tensor$new( NULL,
                             level,
                             private$.dims,
                             private$.type,
                             private$.device )

      transfer( self, .tensor )

      .tensor.ptr <- .tensor$ptrs$tensor
      .tensor.ptr
    },

    .deploy.L0 = function(){
      list( tensor = private$.deploy.tensor( 0L ) )
    },

    .deploy.L1 = function(){
      list( tensor = private$.deploy.tensor( 1L ) )
    },

    .deploy.L2 = function(){
      list( tensor = private$.deploy.tensor( 2L ) )
    },

    .deploy.L3 = function(){
      list( tensor = private$.deploy.tensor( 3L ) )
    },

    .destroy.tensor = function(){
        .Call( "cuR_tensor_destroy",
               private$.ptrs$tensor,
               private$.level,
               private$.type )
    },

    .destroy.L0 = function(){
      return()
    },

    .destroy.L1 = function(){
      private$.destroy.tensor()
    },

    .destroy.L2 = function(){
      private$.destroy.tensor()
    },

    .destroy.L3 = function(){
      private$.destroy.tensor()
    }
  ),

  # active ====
  active = list(
    obj = function( obj ){
      self$check.destroyed()

      if( private$.level != 0L ){
        stop( "Not surfaced, direct object access denied" )
      }

      # Protected
      private$.refs <- TRUE

      if( missing( obj ) ){
        return( private$.ptrs$tensor )
      }else{
        check.obj( obj )
        private$.check.match( obj )

        private$.ptrs$tensor <- obj
        private$.alert.content()
      }
    },

    obj.unsafe = function( obj ){
      self$check.destroyed()

      if( private$.level != 0L ){
        stop( "Not surfaced, direct object access denied" )
      }

      # This access is not registered, the given object
      # will not be protected!!!

      if( missing( obj ) ){
        return( private$.ptrs$tensor )
      }else{
        check.obj( obj )
        private$.check.match( obj )

        private$.ptrs$tensor <- obj
        private$.alert.content()
      }
    },

    dims = function( dims ){
      self$check.destroyed()

      if( missing( dims ) ){
        return( private$.dims )
      }else{
        if( !identical( dims, private$.dims ) ){
          dims <- check.dims( dims )

          if( prod( dims ) != private$.l ){
            stop( "Length mismatch on redim" )
          }

          private$.dims <- dims

          if( private$.level == 0L ){
            self$sever()
            .obj.recut( private$.ptrs$tensor, dims )
          }
        }
      }
    },

    type = function( val ){
      self$check.destroyed()

      if( missing( val ) ){
        return( private$.type )
      }else{
        stop( "Tensor type not directly settable" )
      }
    }
  )
)
