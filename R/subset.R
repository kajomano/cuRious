# These functions are sanity checks and conversion functions for
# tensor (column) subsetting in transfer() and algebraic (cuBLAS) calls

.tensor.subset <- R6Class(
  "tensor.subset",
  public = list(
    off  = NULL,
    ptr  = NULL,

    dims = NULL,
    tens = NULL,

    initialize = function( tens ){
      self$tens = tens
      self$dims = tens$dims
    },

    check.perm = function( perm ){
      if( is.null( perm ) ){
        return()
      }

      if( any( !is.tensor( perm ),
               perm$type != "i",
               self$tens$level == 3,
               perm$level == 3 ) ){
        stop( "Invalid tensor permutation" )
      }

      self$ptr  = perm$ptr
      self$dims = c( self$dims[[1]], perm$l )
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

      self$off  = as.integer( span[[1]] )
      self$dims = c( self$dims[[1]], as.integer( span[[2]] - span[[1]] + 1L ) )
    }
  )
)
