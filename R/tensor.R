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
        if( !is.null( dims ) ){
          stop( "Dims can not be defined when data is supplied" )
        }

        if( !is.null( type ) ){
          stop( "Type can not be defined when data is supplied" )
        }

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
          stop( "Invalid data argument" )
        }
      }

      # TODO ====
      # Redo this

      # Initialize the tensor according to init
      if( is.null( data ) ){
        private$.create.ptr()
        self$clear()
      }else{
        if( copy ){
          if( is.tensor( data ) ){
            if( data$level == 0L && private$.level == 0L ){
              private$.ptr <- data$ptr
              # private$.ptr.acc <- TRUE
            }else{
              private$.create.ptr()
              transfer( data, self )
            }
          }else{
            if( private$.level == 0L ){
              private$.ptr <- data
              # private$.ptr.acc <- TRUE
            }else{
              private$.create.ptr()
              data.tensor <- tensor$new( data )
              transfer( data.tensor, self )
            }
          }
        }else{
          private$.create.ptr()
          self$clear()
        }
      }
    },

    # These functions are only there for objs, are essentially interfaces
    # to R

    # TODO ====
    # Think about
    # private$.ptr.acc <- FALSE
    # for these 2 operations:
    push = function( obj ){
      self$check.destroyed()

      obj <- check.obj( obj )
      private$.match.obj( obj )

      obj.tensor   <- tensor$new( obj, private$.level )
      private$.destroy.ptr()
      private$.ptr <- obj.tensor$ptr

      if( private$.level == 0L ){
        private$.ptr.acc <- TRUE
      }

      invisible( TRUE )
    },

    pull = function(){
      self$check.destroyed()

      tmp <- tensor$new( self, 0L )

      if( private$.level == 0L ){
        private$.ptr.acc <- TRUE
      }

      tmp$ptr
    },

    clear = function(){
      self$sever()

      .Call( paste0("cuR_clear_tensor_", private$.level, "_", private$.type ),
             private$.ptr,
             private$.dims )

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

    .sever = function(){
      if( private$.level == 0L ){
        if( private$.ptr.acc ){
          browser()
          old.obj.ptr <- .Call( "cuR_get_obj_ptr", private$.ptr )

          print( "Severed" )

          # Actual severing
          private$.ptr[[1]] <- private$.ptr[[1]]
          .Call( "cuR_get_obj_ptr", private$.ptr )

          if( !( .Call( "cuR_compare_obj_ptr",
                        .Call( "cuR_get_obj_ptr", private$.ptr ),
                        old.obj.ptr ) ) ){
            print( "Content update alert" )
            private$.alert.content()
          }
        }
        private$.ptr.acc <- FALSE
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

    .create.ptr = function( level = private$.level ){
      if( prod( private$.dims ) > 2^32-1 ){
        # TODO ====
        # Use long int or the correct R type to remove this constraint
        stop( "Object is too large" )
      }

      private$.ptr <- switch(
        as.character( level ),
        `0` = {
          private$.ptr.acc <- FALSE
          obj.create( private$.dims, private$.type )
        },
        `1` = .Call( paste0("cuR_create_tensor_1_", private$.type ), private$.dims ),
        `2` = .Call( paste0("cuR_create_tensor_2_", private$.type ), private$.dims ),
        `3` = {
          .cuda.device.set( private$.device )
          .Call( paste0("cuR_create_tensor_3_", private$.type ), private$.dims )
        }
      )
    },

    .destroy.ptr = function(){
      switch(
        as.character( private$.level ),
        `1` = .Call( paste0("cuR_destroy_tensor_1_", private$.type ), private$.ptr ),
        `2` = .Call( paste0("cuR_destroy_tensor_2_", private$.type ), private$.ptr ),
        `3` = .Call( paste0("cuR_destroy_tensor_3_", private$.type ), private$.ptr )
      )

      private$.ptr <- NULL
    },

    .match.obj = function( obj ){
      if( !identical( obj.dims( obj ), private$.dims ) ){
        stop( "Dims do not match" )
      }
      if( obj.type( obj ) != private$.type ){
        stop( "Types do not match" )
      }
    }
  ),

  active = list(
    ptr = function( val ){
      self$check.destroyed()

      private$.refs <- TRUE

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        if( private$.level != 0L ){
          stop( "Not surfaced, direct tensor access denied" )
        }

        val <- check.obj( val )
        private$.match.obj( val )

        private$.ptr <- val
        private$.alert.content()
      }
    },

    .ptr.unsafe = function( val ){
      self$check.destroyed()

      if( missing( val ) ){
        return( private$.ptr )
      }else{
        stop( "Implement proper transfers" )
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
        tmp <- tensor$new( self, level )

        # Free old memory
        private$.destroy.ptr()

        # Update
        private$.ptr   <- tmp$ptr
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

        private$.device <- device

        if( private$.level == 3L ){
          # Create a placeholder and copy
          tmp <- tensor$new( self, 3L, device = device )

          # Free old memory
          private$.destroy.ptr()

          # Update
          private$.ptr <- tmp$ptr

          private$.alert()
        }else{
          private$.alert.context()
        }
      }
    },

    is.destroyed = function( val ){
      if( missing( val ) ) return( is.null( private$.ptr ) )
    }
  )
)
