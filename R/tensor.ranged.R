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
    initialize = function( tensor, ranges = NULL ){
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

      private$.dims <- c( tensor$dims, rep( 1L, times = .max.array.rank - length( tensor$dims ) ) )

      # Ranges
      if( is.null( ranges ) ){
        private$.ranges <- lapply( 1:.max.array.rank, function( rank ){
          c( 1L, private$.dims[[rank]] )
        })
      }else{
        ranges <- check.ranges( ranges )

        private$.ranges <- lapply( 1:.max.array.rank, function( rank ){
          if( length( ranges ) < rank ){
            range <- NULL
          }else{
            range <- ranges[[rank]]
          }

          dim <- private$.dims[[rank]]

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
      private$.offs <- sapply( 1:.max.array.rank, function( rank ){
        private$.ranges[[rank]][[1]] - 1L
      })

      # Spans
      private$.spans <- sapply( 1:.max.array.rank, function( rank ){
        private$.ranges[[rank]][[2]] - private$.ranges[[rank]][[1]] + 1L
      })

      # Rank for now
      # Last position where dim > 1
      # private$.rank <- which.max( private$.dims > 1 )
      private$.rank <- .max.array.rank
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
    .spans  = NULL,

    # Function set
    .rank   = NULL
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

    rank = function( rank ){
      if( missing( rank ) ){
        return( private$.rank )
      }else{
        rank <- check.rank( rank )

        if( private$.rank > rank ){
          if( which.max( private$.dims > 1 ) > rank ){
            stop( "Tensor rank is bigger than required" )
          }

          private$.dims   <- private$.dims[ 1:rank ]
          private$.ranges <- private$.ranges[ 1:rank ]
          private$.offs   <- private$.offs[ 1:rank ]
          private$.spans  <- private$.spans[ 1:rank ]
          private$.rank   <- rank

        }else if( private$.rank < rank ){
          pad <- ( rank - private$.rank )

          private$.dims   <- c( private$.dims, rep( 1L, times = pad ) )
          private$.ranges <- c( private$.ranges, lapply( 1:pad, function( ... ){ c( 1, 1 ) } ) )
          private$.offs   <- c( private$.offs, rep( 0L, times = pad ) )
          private$.spans  <- c( private$.spans, rep( 1L, times = pad ) )
          private$.rank   <- rank
        }
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
