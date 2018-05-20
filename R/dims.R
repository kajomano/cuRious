# These functions are sanity checks and conversion functions for
# tensor dimensions and (column) subsetting for
# tunnels and algebraic (cuBLAS) calls

.tensor.dims <- R6Class(
  "cuR.tensor.dims",
  public = list(
    span.off = NULL,
    dims     = NULL,

    initialize = function( tens ){
      check.tensor( tens )
      self$dims <- tens$dims
    },

    check.perm = function( perm ){
      if( is.null( perm ) ){
        return()
      }
      check.tensor( perm )

      if( perm$type != "i" ){
        stop( "Invalid tensor permutation" )
      }

      self$dims <- c( self$dims[[1]], perm$l )
    },

    check.span = function( span ){
      if( is.null( span ) ){
        return()
      }

      if( any( !is.obj( span ),
               !is.numeric( span ),
               !length( span ) == 2,
               as.logical( span %% 1 ),
               span[[2]] > self$dims[[2]],
               span[[2]] < span[[1]],
               span[[1]] < 0 ) ){
        stop( "Invalid tensor span" )
      }

      self$span.off <- as.integer( span[[1]] )
      self$dims     <- c( self$dims[[1]], as.integer( span[[2]] - span[[1]] + 1L ) )
    },

    check.trans = function( trans ){
      if( trans ){
        self$dims <- rev( self$dims )
      }
    },

    check.vect = function(){
      if( self$dims[[1]] == 1L ){
        stop( "Tensor is not vector" )
      }
    }
  )
)
