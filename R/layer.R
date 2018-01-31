# Layer classes

# Input layer ====
# Even though this class is the input layer by itself, it is also the super to
# all other layer classes.
layer.input <- R6Class(
  "layer.input",
  public = list(
    height.in  = NULL,
    height.out = NULL,

    nn         = NULL,
    nn.pos     = NULL,

    # Tensors
    tensor.out = NULL,

    initialize = function( height.in, height.out = NULL ){
      private$height.in  <- height.in
      private$height.out <- height.in
    },

    init = function(){},

    # Priming decides what exactly forward and backward passes will do
    deploy.test  = function( batch.size, ... ){

    },

    deploy.train = function( batch.size, ... ){
      self$deploy.test( batch.size, ... )
    },

    retract = function(){},

    forward.pass = function( input ){

    },

    backward.pass = function( error ){
      error
    },
    update.pass = function( ... ){}
  )
)



is.layer <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "layer.input" %in% class( obj )
  })
}

check.layer <- function( ... ){
  if( !all( is.layer( ... ) ) ){
    stop( "Not all objects are layers" )
  }
}
