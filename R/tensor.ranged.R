# This object is a wrapper to tensors. It serves two purposes:
# 1) Users can wrap tensors into tensor.span objects, and set spans for fusions
# 2) Tensor span wrappers check for valid spans, and can calculate strides
#    and other necessary values for fusions

# TODO ====
# Rename to tensor.ranged
# span is span.dims
# Current span is range, does not need to be stored
# Make it possible to overwrite dims here too
# Make it possible to overwrite rank here
# Fucntions should force a certain rank

is.tensor.ranged <- function( tensor.ranged  ){
  "cuR.tensor.ranged" %in% class( tensor.ranged )
}

check.tensor.ranged <- function( tensor.ranged ){
  if( !is.tensor.ranged( tensor.ranged ) ) stop( "Not a ranged tensor" )
}

# Tensor ranged class ====
tensor.ranged <- R6Class(
  "cuR.tensor.ranged",
  public = list(
    initialize = function( tensor, ranges = NULL, rank = length( tensor$dims ) ){
      if( is.tensor( tensor ) ){
        private$.tensor <- tensor
      }else if( is.tensor.ranged( tensor ) ){
        private$.tensor <- tensor$tensor

        if( is.null( ranges ) ){
          ranges <- tensor$ranges
        }
      }else{
        stop( "Invalid (ranged) tensor argument" )
      }

      rank <- check.rank( rank )

      if( max( which( tensor$dims > 1 ) ) > rank ){
        stop( "True tensor rank is bigger than rank given" )
      }

      if( length( tensor$dims ) > rank ){
        private$.dims <- tensor$dims[ 1:rank ]
      }else{
        private$.dims <- c( tensor$dims, rep( 1L, times = rank - length( tensor$dims ) ) )
      }

      # Ranges
      if( is.null( ranges ) ){
        private$.ranges <- lapply( 1:rank, function( r ){
          c( 1L, private$.dims[[r]] )
        })
      }else{
        ranges <- check.ranges( ranges )

        if( length( ranges ) > rank ){
          for( r in ( rank + 1 ):length( ranges ) ){
            range <- ranges[[r]]
            if( !is.null( range ) ){
              if( !identical( range, c( 1L, 1L ) ) ){
                stop( "Mismatch in ranges and rank" )
              }
            }
          }
        }

        private$.ranges <- lapply( 1:rank, function( r ){
          if( length( ranges ) < r ){
            range <- NULL
          }else{
            range <- ranges[[r]]
          }

          dim <- private$.dims[[r]]

          if( is.null( range ) ){
            range <- c( 1L, dim )
          }else{
            # No need to check the other possibilities, as check.ranges
            # establishes that range[[1]] < range[[2]]
            if( range[[1]] < 1L ||
                range[[2]] > dim ){
              stop( "Range out of dims" )
            }
          }

          range
        })
      }

      # Offs
      private$.offs <- sapply( 1:rank, function( r ){
        private$.ranges[[r]][[1]] - 1L
      })

      # Spans
      private$.spans <- sapply( 1:rank, function( r ){
        private$.ranges[[r]][[2]] - private$.ranges[[r]][[1]] + 1L
      })
    }
  ),

  private = list(
    # User set (implicit)
    .tensor = NULL,
    .dims   = NULL,

    # User set (explicit)
    .ranges = NULL,

    # Calculated
    .offs   = NULL,
    .spans  = NULL
  ),

  active = list(
    tensor = function( tensor ){
      if( missing( tensor ) ){
        return( private$.tensor )
      }else{
        stop( "Tensor is not directly settable" )
      }
    },

    dims = function( dims ){
      if( missing( dims ) ){
        return( private$.dims )
      }else{
        stop( "Tensor dimensions are not directly settable" )
      }
    },

    ranges = function( ranges ){
      if( missing( ranges ) ){
        return( private$.ranges )
      }else{
        stop( "Tensor ranges are not directly settable" )
      }
    },

    spans = function( spans ){
      if( missing( spans ) ){
        return( private$.spans )
      }else{
        stop( "Tensor spans are not directly settable" )
      }
    },

    offs = function( offs ){
      if( missing( offs ) ){
        return( private$.offs )
      }else{
        stop( "Tensor offsets are not directly settable" )
      }
    },

    # Wrap is a summary of all the settings in a nice format
    wrap = function( wrap ){
      if( missing( wrap ) ){
        return(
          matrix(
            data     = c( private$.dims,
                          private$.offs,
                          private$.spans ),
            ncol     = 3L,
            dimnames = list( NULL, c( "dims", "offs", "spans" ) )
          )
        )
      }else{
        stop( "Tensor wrap is not directly settable" )
      }
    }
  )
)
