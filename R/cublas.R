# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is written so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas calls (through the whole session) is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cuR.cublas.handle",
  inherit = context,
  private = list(
    .activate = function(){
      .Call( "cuR_cublas_handle_create" )
    },

    .deactivate = function(){
      .Call( "cuR_cublas_handle_destroy", private$.ptr )
    }
  )
)

# cuBLAS linear algebra operations ====
# Parent fusion ====
.cublas.fusion <- R6Class(
  "cuR.cublas.fusion",
  inherit = fusion,
  public = list(
    initialize = function( handle, stream ){
      if( !is.null( handle ) ){
        check.cublas.handle( handle )
      }
      private$.add.ep( handle, "handle" )

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

      if( under ){
        handle <- private$.eps$handle

        if( is.null( handle ) ){
          stop( "Subroutine requires an active cublas handle" )
        }else{
          if( !handle$is.active ){
            stop( "Subroutine requires an active cublas handle" )
          }

          if( handle$device != device ){
            stop( "Cublas handle is not on the correct device" )
          }
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

# TODO ====
# Add sswap from cuBLAS!

# TODO ====
# Add sscal from cuBLAS!

# TODO ====
# Add sgeam from cuBLAS, remove saxpy!

# TODO ====
# Add sdgmm from cuBLAS!

# TODO ====
# L0 calls have wrong subsets if a supposed matrix is actually a vector

# sgemv ====
# y.span(y) = alpha*A.tp(A.span(A)) %*% x.span(x) + beta*y.span(y)
# tp = transpose
cublas.sgemv <- R6Class(
  "cuR.cublas.sgemv",
  inherit = .cublas.fusion,
  public = list(
    initialize = function( A,
                           x,
                           y,
                           A.span = NULL,
                           x.span = NULL,
                           y.span = NULL,
                           A.tp   = FALSE,
                           alpha  = 1,
                           beta   = 1,
                           handle = NULL,
                           stream = NULL  ){
      # Sanity checks
      check.tensor( A )
      check.tensor( x )
      check.tensor( y )

      if( !all( c( x$type == "n", y$type == "n", A$type == "n" ) ) ){
        stop( "All input tensors need to be numeric" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      x.dims <- .tensor.dims$new( x )
      y.dims <- .tensor.dims$new( y )

      x.dims$check.vect()
      y.dims$check.vect()

      A.dims$check.span( A.span )
      x.dims$check.span( x.span )
      y.dims$check.span( y.span )

      if( x.dims$dims[[2]] != A.dims$check.trans( A.tp )[[2]] ||
          y.dims$dims[[2]] != A.dims$check.trans( A.tp )[[1]] ){
        stop( "Not all tensors have matching dimensions" )
      }

      if( !is.numeric( alpha ) || !( length( alpha ) == 1L ) ){
        stop( "Invalid alpha parameter" )
      }

      if( !is.numeric( beta ) || !( length( beta ) == 1L ) ){
        stop( "Invalid beta parameter" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( x, "x" )
      private$.add.ep( y, "y", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$x.span.off <- x.dims$span.off
      private$.params$y.span.off <- y.dims$span.off
      private$.params$A.span.off <- A.dims$span.off

      private$.params$A.tp  <- as.logical( A.tp )

      private$.params$alpha <- as.numeric( alpha )
      private$.params$beta  <- as.numeric( beta )

      super$initialize( handle, stream )
    }
  ),

  private = list(
    .L3.call = function( A.ptr,
                         x.ptr,
                         y.ptr,
                         A.dims,
                         A.span.off = NULL,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.tp,
                         alpha,
                         beta,
                         handle.ptr,
                         stream.ptr = NULL ){

      .Call( "cuR_cublas_sgemv",
             A.ptr,
             x.ptr,
             y.ptr,
             A.dims,
             A.span.off,
             x.span.off,
             y.span.off,
             A.tp,
             alpha,
             beta,
             handle.ptr,
             stream.ptr )

      invisible( TRUE )
    },

    .L0.call = function( A.ptr,
                         x.ptr,
                         y.ptr,
                         A.dims,
                         A.span.off = NULL,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.tp,
                         alpha,
                         beta,
                         handle.ptr = NULL,
                         stream.ptr = NULL ){

      if( !is.null( A.span.off ) ){
        A.ptr <- A.ptr[, A.span.off:( A.span.off + A.dims[[2]] - 1 ) ]
      }

      if( A.tp ){
        A.ptr  <- t( A.ptr )
        A.dims <- rev( A.dims )
      }

      if( !is.null( x.span.off ) ){
        x.ptr <- x.ptr[ x.span.off:( x.span.off + A.dims[[2]] - 1 ) ]
      }

      if( is.null( y.span.off ) ){
        y.range <- 1:A.dims[[1]]
      }else{
        y.range <- y.span.off:( y.span.off + A.dims[[1]] - 1 )
        y.ptr <- y.ptr[ y.range ]
      }

      res <- ( alpha * A.ptr ) %*% x.ptr + ( beta * y.ptr )
      private$.eps.out$y$obj[ y.range ] <- res

      invisible( TRUE )
    }
  )
)

# sger ====
# A.span(A) = alpha*x.span(x) %*% tp(y.span(y)) + A.span(A)
# tp = transpose
cublas.sger <- R6Class(
  "cuR.cublas.sger",
  inherit = .cublas.fusion,
  public = list(
    initialize = function( x,
                           y,
                           A,
                           x.span = NULL,
                           y.span = NULL,
                           A.span = NULL,
                           alpha  = 1,
                           handle = NULL,
                           stream = NULL  ){
      # Sanity checks
      check.tensor( x )
      check.tensor( y )
      check.tensor( A )

      if( !all( c( x$type == "n", y$type == "n", A$type == "n" ) ) ){
        stop( "All input tensors need to be numeric" )
      }

      # Dim checks
      x.dims <- .tensor.dims$new( x )
      y.dims <- .tensor.dims$new( y )
      A.dims <- .tensor.dims$new( A )

      x.dims$check.vect()
      y.dims$check.vect()

      x.dims$check.span( x.span )
      y.dims$check.span( y.span )
      A.dims$check.span( A.span )

      if( x.dims$dims[[2]] != A.dims$dims[[1]] ||
          y.dims$dims[[2]] != A.dims$dims[[2]] ){
        stop( "Not all tensors have matching dimensions" )
      }

      if( !is.numeric( alpha ) || !( length( alpha ) == 1L ) ){
        stop( "Invalid alpha parameter" )
      }

      # Assignments
      private$.add.ep( x, "x" )
      private$.add.ep( y, "y" )
      private$.add.ep( A, "A", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$x.span.off <- x.dims$span.off
      private$.params$y.span.off <- y.dims$span.off
      private$.params$A.span.off <- A.dims$span.off

      private$.params$alpha <- as.numeric( alpha )

      super$initialize( handle, stream )
    }
  ),

  private = list(
    .L3.call = function( x.ptr,
                         y.ptr,
                         A.ptr,
                         A.dims,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.span.off = NULL,
                         alpha,
                         handle.ptr,
                         stream.ptr = NULL ){

      .Call( "cuR_cublas_sger",
             x.ptr,
             y.ptr,
             A.ptr,
             A.dims,
             x.span.off,
             y.span.off,
             A.span.off,
             alpha,
             handle.ptr,
             stream.ptr )

      invisible( TRUE )
    },

    .L0.call = function( x.ptr,
                         y.ptr,
                         A.ptr,
                         A.dims,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.span.off = NULL,
                         alpha,
                         handle.ptr = NULL,
                         stream.ptr = NULL ){

      if( !is.null( x.span.off ) ){
        x.ptr <- x.ptr[ x.span.off:( x.span.off + A.dims[[1]] - 1 ) ]
      }

      if( !is.null( y.span.off ) ){
        y.ptr <- y.ptr[ y.span.off:( y.span.off + A.dims[[2]] - 1 ) ]
      }

      if( is.null( A.span.off ) ){
        A.range <- 1:A.dims[[2]]
      }else{
        A.range <- A.span.off:( A.span.off + A.dims[[2]] - 1 )
        A.ptr <- A.ptr[, A.range ]
      }

      res <- ( alpha * x.ptr ) %*% t( y.ptr ) + A.ptr
      private$.eps.out$A$obj[, A.range ] <- res

      invisible( TRUE )
    }
  )
)

# sgemm ====
# C.span(C) = alpha*A.tp(A.span(A)) %*% B.tp(B.span(B)) + beta*(C.span(C))
# tp = transpose
cublas.sgemm <- R6Class(
  "cuR.cublas.sgemm",
  inherit = .cublas.fusion,
  public = list(
    initialize = function( A,
                           B,
                           C,
                           A.span = NULL,
                           B.span = NULL,
                           C.span = NULL,
                           A.tp   = FALSE,
                           B.tp   = FALSE,
                           alpha  = 1,
                           beta   = 1,
                           handle = NULL,
                           stream = NULL  ){
      # Sanity checks
      check.tensor( A )
      check.tensor( B )
      check.tensor( C )

      if( !all( c( A$type == "n", B$type == "n", C$type == "n" ) ) ){
        stop( "Not all input tensors are numeric" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      B.dims <- .tensor.dims$new( B )
      C.dims <- .tensor.dims$new( C )

      A.dims$check.span( A.span )
      B.dims$check.span( B.span )
      C.dims$check.span( C.span )

      if( !is.logical( A.tp ) || !( length( A.tp ) == 1L ) ){
        stop( "Invalid transpose parameter" )
      }

      if( !is.logical( B.tp ) || !( length( B.tp ) == 1L ) ){
        stop( "Invalid transpose parameter" )
      }

      if( A.dims$check.trans( A.tp )[[2]] != B.dims$check.trans( B.tp )[[1]] ||
          B.dims$check.trans( B.tp )[[2]] != C.dims$dims[[2]] ||
          A.dims$check.trans( A.tp )[[1]] != C.dims$dims[[1]] ){
        stop( "Not all tensors have matching dimensions" )
      }

      if( !is.numeric( alpha ) || !( length( alpha ) == 1L ) ){
        stop( "Invalid alpha parameter" )
      }

      if( !is.numeric( beta ) || !( length( beta ) == 1L ) ){
        stop( "Invalid beta parameter" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( B, "B" )
      private$.add.ep( C, "C", TRUE )

      private$.params$A.dims <- A.dims$dims
      private$.params$B.dims <- B.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$B.span.off <- B.dims$span.off
      private$.params$C.span.off <- C.dims$span.off

      private$.params$A.tp  <- as.logical( A.tp )
      private$.params$B.tp  <- as.logical( B.tp )

      private$.params$alpha <- as.numeric( alpha )
      private$.params$beta  <- as.numeric( beta )

      super$initialize( handle, stream )
    }
  ),

  private = list(
    .L3.call = function( A.ptr,
                         B.ptr,
                         C.ptr,
                         A.dims,
                         B.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         C.span.off = NULL,
                         A.tp,
                         B.tp,
                         alpha,
                         beta,
                         handle.ptr,
                         stream.ptr = NULL ){

      .Call( "cuR_cublas_sgemm",
             A.ptr,
             B.ptr,
             C.ptr,
             A.dims,
             B.dims,
             A.span.off,
             B.span.off,
             C.span.off,
             A.tp,
             B.tp,
             alpha,
             beta,
             handle.ptr,
             stream.ptr )

      invisible( TRUE )
    },

    .L0.call = function( A.ptr,
                         B.ptr,
                         C.ptr,
                         A.dims,
                         B.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         C.span.off = NULL,
                         A.tp,
                         B.tp,
                         alpha,
                         beta,
                         handle.ptr = NULL,
                         stream.ptr = NULL ){

      if( !is.null( A.span.off ) ){
        A.ptr <- A.ptr[, A.span.off:( A.span.off + A.dims[[2]] - 1 ) ]
      }

      if( A.tp ){
        A.ptr  <- t( A.ptr )
      }

      if( !is.null( B.span.off ) ){
        B.ptr <- B.ptr[, B.span.off:( B.span.off + B.dims[[2]] - 1 ) ]
      }

      if( B.tp ){
        B.ptr  <- t( B.ptr )
        B.dims <- rev( B.dims )
      }

      if( is.null( C.span.off ) ){
        C.range <- 1:B.dims[[2]]
      }else{
        C.range <- C.span.off:( C.span.off + B.dims[[2]] - 1 )
        C.ptr   <- C.ptr[, C.range ]
      }

      # Operation
      res <- ( alpha * A.ptr ) %*% B.ptr + ( beta * C.ptr )
      private$.eps.out$C$obj[, C.range ] <- res

      invisible( TRUE )
    }
  )
)
