# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is written so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas calls (through the whole session) is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cuR.cublas.handle",
  inherit = alert.send,
  public = list(
    initialize = function( active = T ){
      if( active ){
        self$activate()
      }
    },

    activate = function(){
      if( is.null( private$.handle ) ){
        private$.handle <- .Call( "cuR_create_cublas_handle" )
        private$.alert()
      }else{
        warning( "cuBLAS handle is already activated" )
      }

      invisible( self )
    },

    deactivate = function(){
      if( !is.null( private$.handle ) ){
        .Call( "cuR_destroy_cublas_handle", private$.handle )
        private$.handle <- NULL
        private$.alert()
      }else{
        warning( "cuBLAS handle is not yet activated" )
      }

      invisible( self )
    }
  ),

  private = list(
    .handle = NULL
  ),

  active = list(
    handle = function( val ){
      if( missing( val ) ) return( private$.handle )
    },

    is.active = function( val ){
      if( missing( val ) ) return( is.null( private$.handle ) )
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

# sger ====
# A = a*x %*% tp(y) + A
# tp = transpose
# x and y are column vectors here, but it does not actually matter, the data
# stored behind x or tp(x) is the same, hence the rows.x argument name
cublas.sger <- R6Class(
  "cuR.cublas.sger",
  inherit = fusion,
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

      if( !is.null( x.span ) ){
        x.dims$check.span( x.span )
      }

      if( !is.null( y.span ) ){
        y.dims$check.span( y.span )
      }

      if( !is.null( A.span ) ){
        A.dims$check.span( A.span )
      }

      if( x.dims$dims[[2]] != A.dims$dims[[1]] ||
          y.dims$dims[[2]] != A.dims$dims[[2]] ){
        stop( "Not all tensors have matching dimensions" )
      }

      if( !is.numeric( alpha ) || !( length( alpha ) == 1L ) ){
        stop( "Invalid alpha parameter" )
      }

      # Assignments
      private$.eps.fix$x <- x
      private$.eps.fix$y <- y
      private$.eps.fix$A <- A

      private$.params$dims <- A.dims$dims

      private$.params$x.span.off <- x.dims$span.off
      private$.params$y.span.off <- y.dims$span.off
      private$.params$A.span.off <- A.dims$span.off

      private$.params$alpha <- as.numeric( alpha )

      if( !is.null( handle ) ){
        check.cublas.handle( handle )
      }
      private$.eps.opt$handle <- handle

      if( !is.null( stream ) ){
        check.cuda.stream( stream )
      }
      private$.eps.opt$stream <- stream

      super$initialize()
    }
  ),

  private = list(
    .L3.call = function( x.ptr,
                          y.ptr,
                          A.ptr,
                          dims,
                          x.span.off = NULL,
                          y.span.off = NULL,
                          A.span.off = NULL,
                          alpha,
                          handle.ptr,
                          stream.ptr = NULL ){

      ret <- .Call( "cuR_cublas_sger",
                    x.ptr,
                    y.ptr,
                    A.ptr,
                    dims,
                    x.span.off,
                    y.span.off,
                    A.span.off,
                    alpha,
                    handle.ptr,
                    stream.ptr )

      if( is.null( ret ) ) stop( "Subroutine failed" )

      invisible( TRUE )
    },

    .L0.call = function( x.ptr,
                         y.ptr,
                         A.ptr,
                         dims,
                         x.span.off = NULL,
                         y.span.off = NULL,
                         A.span.off = NULL,
                         alpha,
                         handle.ptr = NULL,
                         stream.ptr = NULL ){

      if( !is.null( x.span.off ) ){
        x.ptr <- x.ptr[ x.span.off:( x.span.off + dims[[1]] - 1 ) ]
      }

      if( !is.null( y.span.off ) ){
        y.ptr <- y.ptr[ y.span.off:( y.span.off + dims[[2]] - 1 ) ]
      }

      if( is.null( A.span.off ) ){
        A.range <- 1:dims[[2]]
      }else{
        A.range <- A.span.off:( A.span.off + dims[[2]] - 1 )
        A.ptr <- A.ptr[, A.range ]
      }

      private$.eps.fix$A$ptr[, A.range ] <-
        ( alpha * x.ptr ) %*% t( y.ptr ) + A.ptr

      invisible( TRUE )
    },

    .update = function(){
      x <- private$.eps.fix$x
      y <- private$.eps.fix$y
      A <- private$.eps.fix$A

      if( !all( c( x$is.level( 0L ), y$is.level( 0L ), A$is.level( 0L ) ) ) &&
          !all( c( x$is.level( 3L ), y$is.level( 3L ), A$is.level( 3L ) ) ) ){
        stop( "All input tensors need to be on L0 or L3" )
      }

      under <- ( x$level == 3L )

      if( under ){
        if( is.null( private$.eps.opt$handle ) ){
          stop( "Subroutine requires an active cublas handle" )
        }else{
          if( is.null( private$.eps.opt$handle$handle ) ){
            stop( "Subroutine requires an active cublas handle" )
          }
        }

        private$.params$handle.ptr <- private$.eps.opt$handle$handle
      }

      if( !is.null( private$.eps.opt$stream ) ){
        if( !is.null( private$.eps.opt$stream$stream ) ){
          if( !under ){
            warning( "An active stream is given to a synchronous transfer" )
          }

          private$.params$stream.ptr <- private$.eps.opt$stream$stream
        }
      }

      private$.params$x.ptr <- x$ptr
      private$.params$y.ptr <- y$ptr
      private$.params$A.ptr <- A$ptr

      if( under ){
        private$.fun <- private$.L3.call
      }else{
        private$.fun <- private$.L0.call
      }

      super$.update()
    }
  )
)

# sgemm ====
# cols.C(C) = alpha*tp.a(cols.A(A)) %*% tp.b(cols.B(B)) + beta*(cols.C(C))
# tp = transpose
cublas.sgemm <- R6Class(
  "cuR.cublas.sgemm",
  inherit = fusion,
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
        stop( "All input tensors need to be numeric" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      B.dims <- .tensor.dims$new( B )
      C.dims <- .tensor.dims$new( C )

      if( !is.null( A.span ) ){
        A.dims$check.span( A.span )
      }

      if( !is.null( B.span ) ){
        B.dims$check.span( B.span )
      }

      if( !is.null( C.span ) ){
        C.dims$check.span( C.span )
      }

      if( A.dims$check.trans( A.tp )[[2]] != B.dims$check.trans( B.tp )[[1]] ||
          B.dims$check.trans( B.tp )[[2]] != C.dims$dims[[2]] ||
          A.dims$check.trans( A.tp )[[1]] != C.dims$dims[[1]] ){
        stop( "Not all tensors have matching dimensions" )
      }

      if( !is.logical( A.tp ) || !( length( A.tp ) == 1L ) ){
        stop( "Invalid transpose parameter" )
      }

      if( !is.logical( B.tp ) || !( length( B.tp ) == 1L ) ){
        stop( "Invalid transpose parameter" )
      }

      if( !is.numeric( alpha ) || !( length( alpha ) == 1L ) ){
        stop( "Invalid alpha parameter" )
      }

      if( !is.numeric( beta ) || !( length( beta ) == 1L ) ){
        stop( "Invalid beta parameter" )
      }

      # Assignments
      private$.eps.fix$A <- A
      private$.eps.fix$B <- B
      private$.eps.fix$C <- C

      private$.params$A.dims <- A.dims$dims
      private$.params$B.dims <- B.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$B.span.off <- B.dims$span.off
      private$.params$C.span.off <- C.dims$span.off

      private$.params$A.tp <- as.logical( A.tp )
      private$.params$B.tp <- as.logical( B.tp )

      private$.params$alpha <- as.numeric( alpha )
      private$.params$beta  <- as.numeric( beta )

      if( !is.null( handle ) ){
        check.cublas.handle( handle )
      }
      private$.eps.opt$handle <- handle

      if( !is.null( stream ) ){
        check.cuda.stream( stream )
      }
      private$.eps.opt$stream <- stream

      super$initialize()
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

      ret <- .Call( "cuR_cublas_sgemm",
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

      if( is.null( ret ) ) stop( "Subroutine failed" )

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

      if( !is.null( B.span.off ) ){
        B.ptr <- B.ptr[, B.span.off:( B.span.off + B.dims[[2]] - 1 ) ]
      }

      if( is.null( C.span.off ) ){
        C.range <- 1:B.dims[[2]]
      }else{
        C.range <- C.span.off:( C.span.off + B.dims[[1]] - 1 )
        C.ptr <- C.ptr[, C.range ]
      }

      if( A.tp ){
        A.ptr <- t( A.ptr )
      }

      if( B.tp ){
        B.ptr <- t( B.ptr )
      }

      private$.eps.fix$C$ptr[, C.range ] <-
        ( alpha * A.ptr ) %*% B.ptr + ( beta * C.ptr )

      invisible( TRUE )
    },

    .update = function(){
      A <- private$.eps.fix$A
      B <- private$.eps.fix$B
      C <- private$.eps.fix$C

      if( !all( c( A$is.level( 0L ), B$is.level( 0L ), C$is.level( 0L ) ) ) &&
          !all( c( A$is.level( 3L ), B$is.level( 3L ), C$is.level( 3L ) ) ) ){
        stop( "All input tensors need to be on L0 or L3" )
      }

      under <- ( A$level == 3L )

      if( under ){
        if( is.null( private$.eps.opt$handle ) ){
          stop( "Subroutine requires an active cublas handle" )
        }else{
          if( is.null( private$.eps.opt$handle$handle ) ){
            stop( "Subroutine requires an active cublas handle" )
          }
        }

        private$.params$handle.ptr <- private$.eps.opt$handle$handle
      }

      if( !is.null( private$.eps.opt$stream ) ){
        if( !is.null( private$.eps.opt$stream$stream ) ){
          if( !under ){
            warning( "An active stream is given to a synchronous transfer" )
          }

          private$.params$stream.ptr <- private$.eps.opt$stream$stream
        }
      }

      private$.params$A.ptr <- A$ptr
      private$.params$B.ptr <- B$ptr
      private$.params$C.ptr <- C$ptr

      if( under ){
        private$.fun <- private$.L3.call
      }else{
        private$.fun <- private$.L0.call
      }

      super$.update()
    }
  )
)

# TODO ====
# Add cols subset to other cublas calls

# # B = alpha*A + B
# # The trick here is that element-wise addition can be done this way also on
# # matrices, even though thats not the intended use
# cublas.saxpy <- function( handle, tens.A, tens.B, alpha = 1, stream = NULL ){
#   check.cublas.handle.active( handle )
#   check.tensor.under( tens.A, tens.B )
#   if( !is.null( stream ) ){
#     check.cuda.stream( stream )
#     stream <- stream$get.stream
#   }
#
#   # Results go into tens.B
#   ret <- .Call( "cuR_cublas_saxpy",
#                 tens.A$get.obj,
#                 tens.B$get.obj,
#                 tens.A$get.l,
#                 alpha,
#                 handle$get.handle,
#                 stream )
#
#   if( is.null( ret ) ) stop( "Subroutine failed" )
#
#   # If no stream is given, make this call blocking
#   if( is.null( stream ) ){
#     cuda.stream.sync.all()
#   }
#
#   invisible( NULL )
# }
