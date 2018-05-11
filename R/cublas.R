# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is written so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas calls (through the whole session) is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cublas.handle",
  public = list(
    initialize = function(){
      private$.handle <- .Call( "cuR_create_cublas_handle" )
    }
  ),

  private = list(
    .handle = NULL,

    check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The handle is destroyed" )
      }
    }
  ),

  active = list(
    handle = function( val ){
      private$check.destroyed()
      if( missing( val ) ) return( private$.handle )
    },

    is.destroyed = function( val ){
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
cublas.sger <- function( x,
                         y,
                         A,
                         x.rows = NULL,
                         y.rows = NULL,
                         A.cols = NULL,
                         alpha  = 1,
                         handle = NULL,
                         stream = NULL ){
  # Sanity checks
  x <- check.tensor( x )
  y <- check.tensor( y )
  A <- check.tensor( A )

  if( !all( c( x$is.under, y$is.under, A$is.under ) ) &&
      !all( c( x$is.surfaced, y$is.surfaced, A$is.surfaced ) ) ){
    stop( "All input tensors need to be on L0 or L3" )
  }

  if( !all( c( x$type == "n", y$type == "n", A$type == "n" ) ) ){
    stop( "All input tensors need to be numeric" )
  }

  if( A$is.under ){
    check.cublas.handle( handle )
    handle <- handle$handle
  }

  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$stream
  }

  # Dims logic
  if( !is.null( x.rows ) ){
    if( is.obj( x.rows ) ){
      x.subs <- column.range( x, x.rows )
    }else{
      stop( "Invalid row subset for x" )
    }
  }else{
    x.subs <- column.empty( x, x.rows )
  }

  if( !is.null( y.rows ) ){
    if( is.obj( y.rows ) ){
      y.subs <- column.range( y, y.rows )
    }else{
      stop( "Invalid row subset for y" )
    }
  }else{
    y.subs <- column.empty( y, y.rows )
  }

  if( !is.null( A.cols ) ){
    if( is.obj( A.cols ) ){
      A.subs <- column.range( A, A.cols )
    }else{
      stop( "Invalid column subset for A" )
    }
  }else{
    A.subs <- column.empty( A, A.cols )
  }

  if( x.subs$dims[[2]] != A.subs$dims[[1]] ||
      y.subs$dims[[2]] != A.subs$dims[[2]] ){
    stop( "Not all tensors have matching dimensions" )
  }

  if( x.subs$dims[[1]] != 1L || y.subs$dims[[1]] != 1L ){
    stop( "x or y is not a vector" )
  }

  if( A$is.under ){
    ret <- .Call( "cuR_cublas_sger",
                  x$get.obj,
                  y$get.obj,
                  A$get.obj,
                  A.subs$dims,
                  x.subs$off,
                  y.subs$off,
                  A.subs$off,
                  alpha,
                  handle,
                  stream )

    if( is.null( ret ) ) stop( "Subroutine failed" )

    # If no stream is given, make this call blocking
    if( is.null( stream ) ){
      cuda.stream.sync.all()
    }
  }else{
    tmp.x <- obj.create( x.subs$dims )
    tmp.y <- obj.create( y.subs$dims )
    tmp.A <- obj.create( A.subs$dims )

    transfer.core( x$ptr, tmp.x, 0L, 0L, "n", x.subs$dims, x.subs$off )
    transfer.core( y$ptr, tmp.y, 0L, 0L, "n", y.subs$dims, y.subs$off )
    transfer.core( A$ptr, tmp.A, 0L, 0L, "n", A.subs$dims, A.subs$off )

    tmp.A <- ( alpha * tmp.x ) %*% t(tmp.y) + tmp.A

    transfer.core( tmp.A, A$ptr, 0L, 0L, "n", A.subs$dims, NULL, A.subs$off )
  }

  invisible( TRUE )
}


# sgemm ====
# cols.C(C) = alpha*tp.a(cols.A(A)) %*% tp.b(cols.B(B)) + beta*(cols.C(C))
# tp = transpose
cublas.sgemm <- function( A,
                          B,
                          C,
                          A.cols = NULL,
                          B.cols = NULL,
                          C.cols = NULL,
                          A.tp   = FALSE,
                          B.tp   = FALSE,
                          alpha  = 1,
                          beta   = 1,
                          handle = NULL,
                          stream = NULL ){
  # Sanity checks
  A <- check.tensor( A )
  B <- check.tensor( B )
  C <- check.tensor( C )

  if( !all( c( A$is.under, B$is.under, C$is.under ) ) &&
      !all( c( A$is.surfaced, B$is.surfaced,C$is.surfaced ) ) ){
    stop( "All input tensors need to be on L0 or L3" )
  }

  if( !all( c( A$type == "n", B$type == "n", C$type == "n" ) ) ){
    stop( "All input tensors need to be numeric" )
  }

  if( A$is.under ){
    check.cublas.handle( handle )
    handle <- handle$handle
  }

  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$stream
  }

  # Dims logic
  if( !is.null( A.cols ) ){
    if( is.obj( A.cols ) ){
      A.subs <- column.range( A, A.cols )
    }else{
      stop( "Invalid column subset for A" )
    }
  }else{
    A.subs <- column.empty( A, A.cols )
  }

  if( !is.null( B.cols ) ){
    if( is.obj( B.cols ) ){
      B.subs <- column.range( B, B.cols )
    }else{
      stop( "Invalid column subset for B" )
    }
  }else{
    B.subs <- column.empty( B, B.cols )
  }

  if( !is.null( C.cols ) ){
    if( is.obj( C.cols ) ){
      C.subs <- column.range( C, C.cols )
    }else{
      stop( "Invalid column subset for C" )
    }
  }else{
    C.subs <- column.empty( C, C.cols )
  }

  dims.check.A <- A.subs$dims
  dims.check.B <- B.subs$dims

  if( A.tp ){
    dims.check.A <- rev( A.subs$dims )
  }

  if( B.tp ){
    dims.check.B <- rev( B.subs$dims )
  }

  if( dims.check.A[[2]] != dims.check.B[[1]] ||
      dims.check.B[[2]] != C.subs$dims[[2]] ||
      dims.check.A[[1]] != C.subs$dims[[1]] ){
    stop( "Not all tensors have matching dimensions" )
  }

  # Actual calls
  # L3
  if( A$is.under ){
    ret <- .Call( "cuR_cublas_sgemm",
                  A$ptr,
                  B$ptr,
                  C$ptr,
                  A.subs$dims,
                  B.subs$dims,
                  A.subs$off,
                  A.subs$off,
                  C.subs$off,
                  A.tp,
                  B.tp,
                  alpha,
                  beta,
                  handle,
                  stream )

    if( is.null( ret ) ) stop( "Subroutine failed" )

    # If no stream is given, make this call blocking
    if( is.null( stream ) ){
      cuda.stream.sync.all()
    }

  # L0 fallback call
  }else{
    tmp.A <- obj.create( A.subs$dims )
    tmp.B <- obj.create( B.subs$dims )
    tmp.C <- obj.create( C.subs$dims )

    transfer.core( A$ptr, tmp.A, 0L, 0L, "n", A.subs$dims, A.subs$off )
    transfer.core( B$ptr, tmp.B, 0L, 0L, "n", B.subs$dims, B.subs$off )
    transfer.core( C$ptr, tmp.C, 0L, 0L, "n", C.subs$dims, C.subs$off )

    if( A.tp ){
      tmp.A <- t(tmp.A)
    }

    if( B.tp ){
      tmp.B <- t(tmp.B)
    }

    # Math
    tmp.C <- ( alpha * tmp.A ) %*% tmp.B + ( beta * tmp.C )

    transfer.core( tmp.C, C$ptr, 0L, 0L, "n", C.subs$dims, NULL, C.subs$off )
  }

  invisible( TRUE )
}

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
