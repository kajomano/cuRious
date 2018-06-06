# .Calls: src/thrust.cpp

# Thrust operations ====
# Parent fusion ====
.thrust.fusion <- R6Class(
  "cuR.thrust.fusion",
  inherit = fusion,
  public = list(
    initialize = function( stream ){
      if( !is.null( stream ) ){
        check.cuda.stream( stream )
      }
      private$.add.ep( stream, "stream" )
    }
  ),

  private = list(
    .update.context = function( ... ){
      tensors <- sapply( private$.eps, is.tensor )
      tensors <- private$.eps[ tensors ]

      if( !all( sapply( tensors, `[[`, "level" ) == 0L ) &&
          !all( sapply( tensors, `[[`, "level" ) == 3L ) ){
        stop( "Not all tensors are on L0 or L3" )
      }

      under  <- ( tensors[[1]]$level == 3L )
      device <- tensors[[1]]$device

      if( under ){
        if( !all( sapply( tensors, `[[`, "device" ) == device ) ){
          stop( "Not all tensors are on the same device" )
        }
      }

      stream <- private$.eps$stream

      if( !is.null( stream ) ){
        if( stream$is.active ){
          if( !under ){
            stop( "An active stream is given to a synchronous cublas call" )
          }else{
            if( stream$device != device ){
              stop( "Stream is not on the correct device" )
            }
          }
        }
      }

      private$.device <- device

      if( under ){
        private$.fun <- private$.L3.call
      }else{
        private$.fun <- private$.L0.call
      }
    }
  )
)

# pow2 ====
# B <- A^2
thrust.pow2 <- R6Class(
  "cuR.thrust.pow2",
  inherit = .thrust.fusion,
  public = list(
    initialize = function( A,
                           B,
                           A.span = NULL,
                           B.span = NULL,
                           stream = NULL  ){
      # Sanity checks
      check.tensor( A )
      check.tensor( B )

      if( !all( c( A$type == "n", B$type == "n" ) ) ){
        stop( "All input tensors need to be numeric" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      B.dims <- .tensor.dims$new( B )

      A.dims$check.span( A.span )
      B.dims$check.span( B.span )

      if( !identical( A.dims$dims, B.dims$dims ) ){
        stop( "Not all tensors have matching dimensions" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( B, "B", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$B.span.off <- B.dims$span.off

      super$initialize( stream )
    }
  ),

  private = list(
    .L3.call = function( A.ptr,
                         B.ptr,
                         A.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         stream.ptr = NULL ){

      .Call( "cuR_thrust_pow2",
             A.ptr,
             B.ptr,
             A.dims,
             A.span.off,
             B.span.off,
             stream.ptr )

      invisible( TRUE )
    },

    .L0.call = function( A.ptr,
                         B.ptr,
                         A.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         stream.ptr = NULL ){

      if( !is.null( A.span.off ) ){
        A.range <- A.span.off:( A.span.off + A.dims[[2]] - 1 )

        if( A.dims[[1]] == 1L ){
          A.ptr <- A.ptr[ A.range ]
        }else{
          A.ptr <- A.ptr[, A.range ]
        }
      }

      if( !is.null( B.span.off ) ){
        B.range <- B.span.off:( B.span.off + A.dims[[2]] - 1 )
      }else{
        B.range <- 1:A.dims[[2]]
      }

      res <- A.ptr ^ 2

      if( A.dims[[1]] == 1L ){
        private$.eps.out$B$obj[ B.range ] <- res
      }else{
        private$.eps.out$B$obj[, B.range ] <- res
      }

      invisible( TRUE )
    }
  )
)

# cmins ====
# Tells which row is the smallest in every column
thrust.cmins <- R6Class(
  "cuR.thrust.cmins",
  inherit = .thrust.fusion,
  public = list(
    initialize = function( A,
                           x,
                           A.span = NULL,
                           x.span = NULL,
                           stream = NULL  ){
      # Sanity checks
      check.tensor( A )
      check.tensor( x )

      if( A$type != "n" ){
        stop( "Input tensors is not numeric" )
      }

      if( x$type != "i" ){
        stop( "Input tensors is not numeric" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      x.dims <- .tensor.dims$new( x )

      A.dims$check.span( A.span )
      x.dims$check.span( B.span )

      x.dims$check.vect()

      # ITT ====

      if( !identical( A.dims$dims, B.dims$dims ) ){
        stop( "Not all tensors have matching dimensions" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( B, "B", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$B.span.off <- B.dims$span.off

      super$initialize( stream )
    }
  ),

  private = list(
    .L3.call = function( A.ptr,
                         B.ptr,
                         A.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         stream.ptr = NULL ){

      .Call( "cuR_thrust_pow2",
             A.ptr,
             B.ptr,
             A.dims,
             A.span.off,
             B.span.off,
             stream.ptr )

      invisible( TRUE )
    },

    .L0.call = function( A.ptr,
                         B.ptr,
                         A.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         stream.ptr = NULL ){

      if( !is.null( A.span.off ) ){
        A.range <- A.span.off:( A.span.off + A.dims[[2]] - 1 )

        if( A.dims[[1]] == 1L ){
          A.ptr <- A.ptr[ A.range ]
        }else{
          A.ptr <- A.ptr[, A.range ]
        }
      }

      if( !is.null( B.span.off ) ){
        B.range <- B.span.off:( B.span.off + A.dims[[2]] - 1 )
      }else{
        B.range <- 1:A.dims[[2]]
      }

      res <- A.ptr ^ 2

      if( A.dims[[1]] == 1L ){
        private$.eps.out$B$obj[ B.range ] <- res
      }else{
        private$.eps.out$B$obj[, B.range ] <- res
      }

      invisible( TRUE )
    }
  )
)
