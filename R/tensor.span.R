# This object is a wrapper to tensors. It serves two purposes:
# 1) Users can wrap tensors into tensor.span objects, and set spans for fusions
# 2) Tensor span wrappers check for valid spans, and can calculate strides
#    and other necessary values for fusions

is.tensor.span <- function( tensor.span  ){
  "cuR.tensor.span" %in% class( tensor.span )
}

check.tensor.span <- function( tensor.span ){
  if( !is.tensor.span( tensor.span ) ) stop( "Not a tensor span" )
  invisible( tensor.span )
}

# Tensor span class ====
tensor.span <- R6Class(
  "cuR.tensor.span",
  public = list(
    initialize = function( tensor, span = NULL ){
      if( is.tensor( tensor ) ){
        private$.tensor <- tensor
      }else if( is.tensor.span( tensor ) ){
        private$.tensor <- tensor$tensor

        if( is.null( span ) ){
          span <- tensor$span
        }
      }else{
        stop( "Invalid tensor argument" )
      }

      private$.dims <- tensor$dims

      # Span
      if( is.null( span ) ){
        private$.span <- lapply( 1:.max.array.rank, function( rank ){
          c( 1L, private$.dims[[rank]] )
        })
      }else{
        span <- check.span( span )

        private$.span <- lapply( 1:.max.array.rank, function( rank ){
          if( length( span ) < rank ){
            range <- NULL
          }else{
            range <- span[[rank]]
          }

          dim <- private$.dims[[rank]]

          if( is.null( range ) ){
            range <- c( 1L, dim )
          }else{
            # No need to check the other possibilities, as check.span
            # establishes that range[[1]] < range[[2]]
            if( range[[1]] < 1L ||
                range[[2]] > dim ){
              stop( "Span out of range" )
            }
          }

          range
        })
      }

      # Span.dims
      private$.span.dims <- sapply( 1:.max.array.rank, function( rank ){
        private$.span[[rank]][[2]] - private$.span[[rank]][[1]] + 1L
      })

      # Span.offs
      private$.span.offs <- sapply( 1:.max.array.rank, function( rank ){
        private$.span[[rank]][[1]] - 1L
      })

      # Rank
      # Last position where dim > 1
      private$.rank <- which.max( private$.span.dims > 1 )
    }

    # check.perm = function( perm ){
    #   if( is.null( perm ) ){
    #     return()
    #   }
    #
    #   check.tensor( perm )
    #
    #   if( perm$type != "i" ||
    #       perm$dims[[1]] != 1L ){
    #     stop( "Invalid tensor permutation" )
    #   }
    #
    #   self$dims.perm <- c( self$dims.orig[[1]], perm$l )
    #   self$dims <- self$dims.perm
    # },
    #
    # check.span = function( span ){
    #   if( is.null( span ) ){
    #     self$span.off <- 1L
    #     return()
    #   }
    #
    #   if( is.null( self$dims.perm ) ){
    #     dims <- self$dims.orig
    #   }else{
    #     dims <- self$dims.perm
    #   }
    #
    #   if( any( !is.obj( span ),
    #            !is.numeric( span ) ) ){
    #     stop( "Invalid tensor span" )
    #   }else if( any( !length( span ) == 2,
    #                  as.logical( span %% 1 ),
    #                  span[[2]] > dims[[2]],
    #                  span[[2]] < span[[1]],
    #                  span[[1]] < 1 ) ){
    #     stop( "Invalid tensor span" )
    #   }
    #
    #   self$span.off <- as.integer( span[[1]] )
    #   self$dims     <- c( dims[[1]], as.integer( span[[2]] - span[[1]] + 1L ) )
    # },
    #
    # check.trans = function( trans ){
    #   if( trans ){
    #     rev( self$dims )
    #   }else{
    #     self$dims
    #   }
    # },
    #
    # check.vect = function(){
    #   if( self$dims[[1]] != 1L ){
    #     stop( "Tensor is not vector" )
    #   }
    # }
  ),

  private = list(
    .tensor    = NULL,
    .dims      = NULL,

    .span      = NULL,
    .span.dims = NULL,
    .span.offs = NULL,

    .rank      = NULL

    # .offsets   = NULL,
    # .strides   = NULL
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
        stop( "Tensor dims are not directly settable" )
      }
    },

    span = function( span ){
      if( missing( span ) ){
        return( private$.span )
      }else{
        stop( "Tensor span is not directly settable" )
      }
    },

    span.dims = function( span.dims ){
      if( missing( span.dims ) ){
        return( private$.span.dims )
      }else{
        stop( "Tensor span dims are not directly settable" )
      }
    },

    span.offs = function( span.offs ){
      if( missing( span.offs ) ){
        return( private$.span.offs )
      }else{
        stop( "Tensor span offs are not directly settable" )
      }
    },

    rank = function( rank ){
      if( missing( rank ) ){
        return( private$.rank )
      }else{
        stop( "Tensor rank is not directly settable" )
      }
    }
  )
)
