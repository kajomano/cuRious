# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is written so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas calls (through the whole session) is advisable.

# cuBLAS handle class ====
cublas.context <- R6Class(
  "cuR.cublas.context",
  inherit = fusion.context,
  private = list(
    .deploy = function(){
      super$.deploy(
        expression(
          list( handle = .Call( "cuR_cublas_handle_create" ) )
        )
      )
    },

    .destroy = function(){
      super$.destroy(
        expression(
          .Call( "cuR_cublas_handle_destroy", private$.ptrs$handle )
        )
      )
    }
  )
)

# cuBLAS linear algebra operations ====
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
  inherit = contexted.fusion,
  public = list(
    initialize = function( A,
                           x,
                           y,
                           A.span  = NULL,
                           x.span  = NULL,
                           y.span  = NULL,
                           A.tp    = FALSE,
                           alpha   = 1,
                           beta    = 1,
                           context = NULL ){
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
        stop( "Tensor dimension mismatch" )
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

      super$initialize( context )
    }
  ),

  private = list(
    .L3.call = function( A.tensor,
                         x.tensor,
                         y.tensor,
                         A.dims,
                         A.span.off = NULL,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.tp,
                         alpha,
                         beta,
                         context.handle,
                         stream.queue  = NULL,
                         stream.stream = NULL ){

      .Call( "cuR_cublas_sgemv",
             A.tensor,
             x.tensor,
             y.tensor,
             A.dims,
             A.span.off,
             x.span.off,
             y.span.off,
             A.tp,
             alpha,
             beta,
             context.handle,
             stream.queue,
             stream.stream )

      invisible( TRUE )
    },

    .L0.call = function( A.tensor,
                         x.tensor,
                         y.tensor,
                         A.dims,
                         A.span.off = NULL,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.tp,
                         alpha,
                         beta,
                         context.handle = NULL,
                         stream.queue   = NULL,
                         stream.stream  = NULL ){

      if( !is.null( A.span.off ) ){
        A.tensor <- A.tensor[, A.span.off:( A.span.off + A.dims[[2]] - 1 ) ]
      }

      if( A.tp ){
        A.tensor <- t( A.tensor )
        A.dims <- rev( A.dims )
      }

      if( !is.null( x.span.off ) ){
        x.tensor <- x.tensor[ x.span.off:( x.span.off + A.dims[[2]] - 1 ) ]
      }

      if( is.null( y.span.off ) ){
        y.range <- 1:A.dims[[1]]
      }else{
        y.range <- y.span.off:( y.span.off + A.dims[[1]] - 1 )
        y.tensor <- y.tensor[ y.range ]
      }

      res <- ( alpha * A.tensor ) %*% x.tensor + ( beta * y.tensor )
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
  inherit = contexted.fusion,
  public = list(
    initialize = function( x,
                           y,
                           A,
                           x.span  = NULL,
                           y.span  = NULL,
                           A.span  = NULL,
                           alpha   = 1,
                           context = NULL ){
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
        stop( "Tensor dimension mismatch" )
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

      super$initialize( context )
    }
  ),

  private = list(
    .L3.call = function( x.tensor,
                         y.tensor,
                         A.tensor,
                         A.dims,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.span.off = NULL,
                         alpha,
                         context.handle,
                         stream.queue  = NULL,
                         stream.stream = NULL ){

      .Call( "cuR_cublas_sger",
             x.tensor,
             y.tensor,
             A.tensor,
             A.dims,
             x.span.off,
             y.span.off,
             A.span.off,
             alpha,
             context.handle,
             stream.queue,
             stream.stream )

      invisible( TRUE )
    },

    .L0.call = function( x.tensor,
                         y.tensor,
                         A.tensor,
                         A.dims,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.span.off = NULL,
                         alpha,
                         context.handle = NULL,
                         stream.queue   = NULL,
                         stream.stream  = NULL ){

      if( !is.null( x.span.off ) ){
        x.tensor <- x.tensor[ x.span.off:( x.span.off + A.dims[[1]] - 1 ) ]
      }

      if( !is.null( y.span.off ) ){
        y.tensor <- y.tensor[ y.span.off:( y.span.off + A.dims[[2]] - 1 ) ]
      }

      if( is.null( A.span.off ) ){
        A.range <- 1:A.dims[[2]]
      }else{
        A.range <- A.span.off:( A.span.off + A.dims[[2]] - 1 )
        A.tensor <- A.tensor[, A.range ]
      }

      res <- ( alpha * x.tensor ) %*% t( y.tensor ) + A.tensor

      if( A.dims[[1]] == 1L ){
        private$.eps.out$A$obj[ A.range ] <- res
      }else{
        private$.eps.out$A$obj[, A.range ] <- res
      }

      invisible( TRUE )
    }
  )
)

# sgemm ====
# C.span(C) = alpha*A.tp(A.span(A)) %*% B.tp(B.span(B)) + beta*(C.span(C))
# tp = transpose
cublas.sgemm <- R6Class(
  "cuR.cublas.sgemm",
  inherit = contexted.fusion,
  public = list(
    initialize = function( A,
                           B,
                           C,
                           A.span  = NULL,
                           B.span  = NULL,
                           C.span  = NULL,
                           A.tp    = FALSE,
                           B.tp    = FALSE,
                           alpha   = 1,
                           beta    = 1,
                           context = NULL ){
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
        stop( "Tensor dimension mismatch" )
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

      super$initialize( context )
    }
  ),

  private = list(
    .L3.call = function( A.tensor,
                         B.tensor,
                         C.tensor,
                         A.dims,
                         B.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         C.span.off = NULL,
                         A.tp,
                         B.tp,
                         alpha,
                         beta,
                         context.handle,
                         stream.queue  = NULL,
                         stream.stream = NULL ){

      .Call( "cuR_cublas_sgemm",
             A.tensor,
             B.tensor,
             C.tensor,
             A.dims,
             B.dims,
             A.span.off,
             B.span.off,
             C.span.off,
             A.tp,
             B.tp,
             alpha,
             beta,
             context.handle,
             stream.queue,
             stream.stream )

      invisible( TRUE )
    },

    .L0.call = function( A.tensor,
                         B.tensor,
                         C.tensor,
                         A.dims,
                         B.dims,
                         A.span.off = NULL,
                         B.span.off = NULL,
                         C.span.off = NULL,
                         A.tp,
                         B.tp,
                         alpha,
                         beta,
                         context.handle = NULL,
                         stream.queue   = NULL,
                         stream.stream  = NULL ){

      if( !is.null( A.span.off ) ){
        A.tensor <- A.tensor[, A.span.off:( A.span.off + A.dims[[2]] - 1 ) ]
      }

      if( A.tp ){
        A.tensor <- t( A.tensor )
        A.dims   <- rev( A.dims )
      }

      if( !is.null( B.span.off ) ){
        B.tensor <- B.tensor[, B.span.off:( B.span.off + B.dims[[2]] - 1 ) ]
      }

      if( B.tp ){
        B.tensor <- t( B.tensor )
        B.dims   <- rev( B.dims )
      }

      if( is.null( C.span.off ) ){
        C.range <- 1:B.dims[[2]]
      }else{
        C.range  <- C.span.off:( C.span.off + B.dims[[2]] - 1 )
        C.tensor <- C.tensor[, C.range ]
      }

      # Operation
      res <- ( alpha * A.tensor ) %*% B.tensor + ( beta * C.tensor )

      if( A.dims[[1]] == 1L ){
        private$.eps.out$C$obj[ C.range ] <- res
      }else{
        private$.eps.out$C$obj[, C.range ] <- res
      }

      invisible( TRUE )
    }
  )
)
