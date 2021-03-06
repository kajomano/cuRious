# These functions are sanity checks and conversion functions for
# tensor dimensions and (column) subsetting for
# tunnels and algebraic (cuBLAS) calls

.tensor.dims <- R6Class(
  "cuR.tensor.dims",
  public = list(
    span.off  = 1L,
    dims.orig = NULL,
    dims.perm = NULL,
    dims      = NULL,

    initialize = function( tens ){
      check.tensor( tens )
      self$dims.orig <- tens$dims
      self$dims      <- self$dims.orig
    },

    check.perm = function( perm ){
      if( is.null( perm ) ){
        return()
      }

      check.tensor( perm )

      if( perm$type != "i" ||
          perm$dims[[1]] != 1L ){
        stop( "Invalid tensor permutation" )
      }

      self$dims.perm <- c( self$dims.orig[[1]], perm$l )
      self$dims <- self$dims.perm
    },

    check.span = function( span ){
      if( is.null( span ) ){
        self$span.off <- 1L
        return()
      }

      if( is.null( self$dims.perm ) ){
        dims <- self$dims.orig
      }else{
        dims <- self$dims.perm
      }

      if( any( !is.obj( span ),
               !is.numeric( span ) ) ){
        stop( "Invalid tensor span" )
      }else if( any( !length( span ) == 2,
                     as.logical( span %% 1 ),
                     span[[2]] > dims[[2]],
                     span[[2]] < span[[1]],
                     span[[1]] < 1 ) ){
        stop( "Invalid tensor span" )
      }

      self$span.off <- as.integer( span[[1]] )
      self$dims     <- c( dims[[1]], as.integer( span[[2]] - span[[1]] + 1L ) )
    },

    check.trans = function( trans ){
      if( trans ){
        rev( self$dims )
      }else{
        self$dims
      }
    },

    check.vect = function(){
      if( self$dims[[1]] != 1L ){
        stop( "Tensor is not vector" )
      }
    }
  )
)
