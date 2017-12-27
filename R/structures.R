# .Calls: src/structures.c

# Test if cuRious math.struct object
is.struct <- function( obj ){
  "math.struct" %in% class(obj)
}

# Test if data is stored on the GPU
is.under <- function( obj ){
  if( is.struct(obj) ){
    obj$is.under
  }else{
    FALSE
  }
}

error.undef.op <- function(){
  stop( "Undefined operation" )
}

# math.struct ====
# Parent structure to any mathematical structure
math.struct <- R6Class(
  "math.struct",
  public = list(
    obj       = NULL,
    initialize = function( obj ){
      # Set correct storage type
      storage.mode( obj ) <- "double"

      # Store object
      self$obj <- obj

      # Set info
      private$under  <- FALSE
    },
    dive    = error.undef.op,
    surface = error.undef.op,
    push    = error.undef.op,
    pull    = error.undef.op
  ),

  private = list(
    under   = NULL
  ),

  active = list(
    is.under  = function( val ){
      if( missing(value) ) return( private$under )
    },
    # l is not necessary the length of a vector, it is the number of
    # elements in the data; this is important for the c pointers
    l = error.undef.op
  )
)

# vect ====
# Test if cuRious vect object
is.vect <- function( obj ){
  "vect" %in% class(obj)
}

# Vector structure
vect <- R6Class(
  "vect",
  inherit = math.struct,
  public = list(
    initialize = function( obj ){
      # Check for correct R type
      if(!is.vector( obj )) stop( "Invalid R object" )

      # Call parent's init
      super$initialize( obj )

      # Set extra info
      private$length <- length( obj )
    },

    dive = function(){
      if( !private$under ){
        private$under <- TRUE
        self$obj <- .Call( "dive_num_vect", self$obj, private$length )
      }
    },

    surface = function(){
      if( private$under ){
        private$under <- FALSE
        self$obj <- .Call( "surface_num_vect", self$obj, private$length )
      }
    },

    push = function( obj ){
      # Check for correct R type, length
      if( !is.vector( obj ) ) error.inv.type()
      if( private$length != length(obj) ) stop( "Dimensions do not match" )

      # Set correct storage type
      storage.mode( obj ) <- "double"

      if( private$under ){
        .Call( "push_num_vect", obj, private$length, self$obj )
      }else{
        # You could theoretically set an object with different dimensions,
        # but it is not allowed
        self$obj <- obj
      }

      invisible( TRUE )
    },

    pull = function(){
      if( private$under ){
        .Call( "surface_num_vect", self$obj, private$length )
      }else{
        self$obj
      }
    }
  ),

  private = list(
    length = NULL
  ),

  active = list(
    l = function( val ){
      if( missing(value) ) return( private$length )
    }
  )
)
