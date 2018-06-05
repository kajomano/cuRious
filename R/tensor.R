# .Calls: src/tensor.cpp
#
# cuRious currently supports 1D and 2D tensors. Scalar values can be
# easily passed to the GPU as function arguments, no need to use the tensor
# framework for them.

# Level 0: R object (        host memory, double, int, int  )
# Level 1: C array  (        host memory, float,  int, bool )
# Level 2: C array  ( pinned host memory, float,  int, bool )
# Level 3: C array  (      device memory, float,  int, bool )

# Tensors implement reference counting if the content is accessed by $ptr

# Tensor class ====
tensor <- R6Class(
  "cuR.tensor",
  inherit = alert.send,
  public = list(
    initialize = function( data   = NULL,
                           level  = NULL,
                           dims   = NULL,
                           type   = NULL,
                           copy   = TRUE,
                           device = NULL
    ){
      # If data is not given
      if( is.null( data ) ){
        if( is.null( dims ) ){
          private$.dims <- c( 1L, 1L )
        }else{
          private$.dims <- check.dims( dims )
        }

        if( is.null( type ) ){
          private$.type <- "n"
        }else{
          private$.type <- check.type( type )
        }

        if( is.null( level ) ){
          private$.level <- 0L
        }else{
          private$.level <- check.level( level )
        }

        if( is.null( device ) ){
          private$.device <- cuda.device.default.get()
        }else{
          private$.device <- check.device( device )
        }

      # If data is supported
      }else{
        if( is.obj( data ) ){
          private$.dims <- obj.dims( data )
          private$.type <- obj.type( data )

          if( is.null( level ) ){
            private$.level <- 0L
          }else{
            private$.level <- check.level( level )
          }

          if( is.null( device ) ){
            private$.device <- cuda.device.default.get()
          }else{
            private$.device <- check.device( device )
          }
        }else if( is.tensor( data ) ){
          private$.dims  <- data$dims
          private$.type  <- data$type

          if( is.null( level ) ){
            private$.level <- data$level
          }else{
            private$.level <- check.level( level )
          }

          if( is.null( device ) ){
            private$.device <- data$device
          }else{
            private$.device <- check.device( device )
          }
        }else{
          stop( "Invalid data format" )
        }

        if( !is.null( dims ) ){
          if( !identical( check.dims( dims ), private$.dims ) ){
            stop( "Data dims and supported dims do not match" )
          }
        }

        if( !is.null( type ) ){
          if( check.type( type ) != private$.type ) ){
            stop( "Data type and supported type does not match" )
          }
        }
      }

      # Initial data
      if( is.null( data ) ){
        private$.ptr <- private$.create.ptr()
        self$clear()
      }else{
        if( copy ){
          if( is.tensor( data ) ){
            if( data$level == 0L && private$.level == 0L ){
              private$.ptr  <- data$obj
              private$.refs <- TRUE
            }else{
              private$.ptr  <- private$.create.ptr()
              private$.push( data )
            }
          }else if( is.obj( data ) ){
            if( private$.level == 0L ){
              private$.ptr  <- data
              private$.refs <- TRUE
            }else{
              private$.ptr  <- private$.create.ptr()
              private$.push( data )
            }
          }
        }else{
          private$.ptr <- private$.create.ptr()
          self$clear()
        }
      }
    },

    # Push takes objects or tensors, pull takes only objects
    # Reason: if you want to pull into a tensor, you can push into it this
    # tensor. If you need a new tensor while also data being pulled into it,
    # create a new tensor with data = this tensor.
    push = function( data ){
      self$check.destroyed()

      private$.match( data )

      self$sever()
      private$.push( data )

      invisible( TRUE )
    },

    pull = function(){
      self$check.destroyed()

      private$.temp( level = 0L )
    },

    clear = function(){
      self$sever()

      .Call( "cuR_tensor_clear",
             private$.ptr,
             private$.level,
             private$.dims,
             private$.type )

      invisible( TRUE )
    },

    destroy = function(){
      self$check.destroyed()
      private$.destroy.ptr()
      private$.alert()
    },

    check.destroyed = function(){
      if( is.null( private$.ptr ) ){
        stop( "The tensor is destroyed" )
      }

      invisible( TRUE )
    },

    sever = function(){
      if( private$.level == 0L ){
        if( private$.refs ){
          private$.ptr  <- .Call( "cuR_object_duplicate", private$.ptr )
          private$.refs <- FALSE
          private$.alert.content()
        }
      }

      invisible( TRUE )
    }
  ),

  private = list(
    .ptr     = NULL,
    .level   = NULL,
    .dims    = NULL,
    .type    = NULL,
    .device  = NULL,

    # Outside references
    .refs    = FALSE,

    .create.ptr = function( level = private$.level, device = private$.device ){
      if( prod( private$.dims ) > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Object is too large" )
      }

      if( !level ){
        obj.create( private$.dims, private$.type )
      }else{
        if( level == 3L ){
          .cuda.device.set( device )
        }

        .Call( "cuR_tensor_create",
               level,
               private$.dims,
               private$.type )
      }
    },

    .destroy.ptr = function(){
      if( private$.level ){
        .Call( "cuR_tensor_destroy",
               private$.ptr,
               private$.level,
               private$.type )
      }

      private$.ptr <- NULL
    },

    .match = function( data ){
      if( is.tensor( data ) ){
        dims <- data$dims
        type <- data$type
      }else if( is.obj( data ) ){
        dims <- obj.dims( data )
        type <- obj.type( data )
      }else{
        stop( "Invalid data format" )
      }

      if( !identical( dims, private$.dims ) ){
        stop( "Dims do not match" )
      }

      if( type != private$.type ){
        stop( "Types do not match" )
      }
    },

    .push = function( data ){
      if( is.tensor( data ) ){
        ptr    <- data$ptr
        level  <- data$level
        device <- data$device
      }else if( is.obj( data ) ){
        ptr    <- data
        level  <- 0L
        device <- 0L
      }else{
        stop( "Invalid data format" )
      }

      .transfer.ptr.choose( level,
                            private$.level,
                            device,
                            private$.device )( ptr,
                                               private$.ptr,
                                               level,
                                               private$.level,
                                               private$.type,
                                               private$.dims )
    },

    .temp = function( level = private$.level, device = private$.device ){
      tmp <- private$.create.ptr( level = level, device = device )
      .transfer.ptr.choose( private$.level,
                            level,
                            private$.device,
                            device )( private$.ptr,
                                      tmp,
                                      private$.level,
                                      level,
                                      private$.type,
                                      private$.dims )
      tmp
    }
  ),

  active = list(
    obj = function( val ){
      self$check.destroyed()

      if( private$.level != 0L ){
        stop( "Not surfaced, direct object access denied" )
      }

      # Protected
      private$.refs <- TRUE

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        val <- check.obj( val )
        private$.match( val )

        private$.ptr <- val
        private$.alert.content()
      }
    },

    ptr = function( val ){
      self$check.destroyed()

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        stop( "Tensor pointer is not directly settable" )
      }
    },

    dims = function( val ){
      self$check.destroyed()
      if( missing( val ) ) return( private$.dims )
    },

    l = function( val ){
      if( missing( val ) ) return( as.integer( prod( self$dims ) ) )
    },

    type = function( val ){
      self$check.destroyed()
      if( missing( val ) ) return( private$.type )
    },

    level = function( level ){
      self$check.destroyed()

      if( missing( level ) ){
        return( private$.level )
      }else{
        level <- check.level( level )

        if( private$.level == level ){
          return()
        }

        # Create a placeholder and copy
        tmp <- private$.temp( level = level )

        # Free old memory
        private$.destroy.ptr()

        # Update
        private$.ptr   <- tmp
        private$.refs  <- FALSE
        private$.level <- level

        # Both context and content changed
        private$.alert()
      }
    },

    device = function( device ){
      self$check.destroyed()

      if( missing( device) ){
        return( private$.device )
      }else{
        device <- check.device( device )

        if( private$.device == device ){
          return()
        }

        if( private$.level == 3L ){
          # Create a placeholder and copy
          tmp <- private$.temp( device = device )

          # Free old memory
          private$.destroy.ptr()

          # Update
          private$.ptr <- tmp

          private$.alert()
        }else{
          private$.alert.context()
        }

        private$.device <- device
      }
    },

    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.ptr ) )
    }
  )
)
