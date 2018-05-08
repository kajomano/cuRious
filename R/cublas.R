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
cublas.sger <- function( tens.x,
                         tens.y,
                         tens.A,
                         rows.x = NULL,
                         rows.y = NULL,
                         cols.A = NULL,
                         alpha  = 1,
                         handle = NULL,
                         stream = NULL ){
  # Sanity checks
  if( !all( is.under( tens.x, tens.y, tens.A ) ) &&
      !all( is.surfaced( tens.x, tens.y, tens.A ) ) ){
    stop( "All inputs need to be on L0 or L3" )
  }

  if( is.under( tens.x ) ){
    check.cublas.handle.active( handle )
  }

  dims.x <- tens.x$get.dims
  dims.y <- tens.y$get.dims
  dims.A <- tens.A$get.dims

  if( dims.x[2] != 1L || dims.y[2] != 1L ){
    stop( "x and y need to be vectors" )
  }

  off.rows.x <- 1L
  off.rows.y <- 1L
  off.cols.A <- 1L

  if( !is.null( rows.x ) ){
    if( !is.list( rows.x ) ){
      stop( "Only row ranges are accepted" )
    }
    off.rows.x  <- as.integer( rows.x[[1]] )
    dims.x[[1]] <- as.integer( rows.x[[2]] - rows.x[[1]] + 1L )
  }

  if( !is.null( rows.y ) ){
    if( !is.list( rows.y ) ){
      stop( "Only row ranges are accepted" )
    }
    off.rows.y  <- as.integer( rows.y[[1]] )
    dims.y[[1]] <- as.integer( rows.y[[2]] - rows.y[[1]] + 1L )
  }

  if( !is.null( cols.A ) ){
    if( !is.list( cols.A ) ){
      stop( "Only column ranges are accepted" )
    }
    off.cols.A  <- as.integer( cols.A[[1]] )
    dims.A[[2]] <- as.integer( cols.A[[2]] - cols.A[[1]] + 1L )
  }

  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$get.stream
  }

  if( dims.x[1] != dims.A[1] ||
      dims.y[1] != dims.A[2] ){
    stop( "Not all tensors have matching dimensions" )
  }

  if( is.under( tens.A ) ){
    ret <- .Call( "cuR_cublas_sger",
                  tens.x$get.obj,
                  tens.y$get.obj,
                  tens.A$get.obj,
                  dims.A,
                  off.rows.x,
                  off.rows.y,
                  off.cols.A,
                  alpha,
                  handle$get.handle,
                  stream )

    if( is.null( ret ) ) stop( "Subroutine failed" )

    # If no stream is given, make this call blocking
    if( is.null( stream ) ){
      cuda.stream.sync.all()
    }
  }else{
    tmp.x <- transfer( tens.x, cols.src = rows.x )
    tmp.y <- transfer( tens.y, cols.src = rows.y )
    tmp.A <- transfer( tens.A, cols.src = cols.A )

    tmp.A <- ( alpha * tmp.x ) %*% t(tmp.y) + tmp.A
    transfer( tmp.A, tens.A, cols.dst = cols.A )
  }

  invisible( TRUE )
}


# sgemm ====
# cols.C(C) = alpha*tp.a(cols.A(A)) %*% tp.b(cols.B(B)) + beta*(cols.C(C))
# tp = transpose
cublas.sgemm <- function( tens.A,
                          tens.B,
                          tens.C,
                          cols.A = NULL,
                          cols.B = NULL,
                          cols.C = NULL,
                          tp.A   = FALSE,
                          tp.B   = FALSE,
                          alpha  = 1,
                          beta   = 1,
                          handle = NULL,
                          stream = NULL ){
  # Sanity checks
  tens.A <- check.tensor( tens.A )
  tens.B <- check.tensor( tens.B )
  tens.C <- check.tensor( tens.C )

  if( !all( c( tens.A$is.under, tens.B$is.under, tens.C$is.under ) ) &&
      !all( c( tens.A$is.surfaced, tens.B$is.surfaced, tens.C$is.surfaced ) ) ){
    stop( "All input tensors need to be on L0 or L3" )
  }

  if( !all( c( tens.A$type == "n", tens.B$type == "n", tens.C$type == "n" ) ) ){
    stop( "All input tensors need to be numeric" )
  }

  if( tens.A$is.under ){
    check.cublas.handle( handle )
    handle <- handle$handle
  }

  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$stream
  }

  # Dims logic
  dims.A <- tens.A$dims
  dims.B <- tens.B$dims
  dims.C <- tens.C$dims

  off.cols.A <- NULL
  off.cols.B <- NULL
  off.cols.C <- NULL

  if( !is.null( cols.A ) ){
    if( !is.numeric( cols.A ) || !length( cols.A ) == 2 ){
      stop( "Only column ranges are accepted" )
    }

    if( as.logical( cols.A %% 1 ) ||
        cols.A[[2]] > tens.A$dims[[2]] ||
        cols.A[[2]] < cols.A[[1]] ||
        cols.A[[1]] < 0 ){
      stop( "Invalid column range for tensor A" )
    }

    off.cols.A  <- as.integer( cols.A[[1]] )
    dims.A[[2]] <- as.integer( cols.A[[2]] - cols.A[[1]] + 1L )
  }

  if( !is.null( cols.B ) ){
    if( !is.numeric( cols.B ) || !length( cols.B ) == 2 ){
      stop( "Only column ranges are accepted" )
    }

    if( as.logical( cols.B %% 1 ) ||
        cols.B[[2]] > tens.B$dims[[2]] ||
        cols.B[[2]] < cols.B[[1]] ||
        cols.B[[1]] < 0 ){
      stop( "Invalid column range for tensor B" )
    }

    off.cols.B  <- as.integer( cols.B[[1]] )
    dims.B[[2]] <- as.integer( cols.B[[2]] - cols.B[[1]] + 1L )
  }

  if( !is.null( cols.C ) ){
    if( !is.numeric( cols.C ) || !length( cols.C ) == 2 ){
      stop( "Only column ranges are accepted" )
    }

    if( as.logical( cols.C %% 1 ) ||
        cols.C[[2]] > tens.C$dims[[2]] ||
        cols.C[[2]] < cols.C[[1]] ||
        cols.C[[1]] < 0 ){
      stop( "Invalid column range for tensor C" )
    }

    off.cols.C  <- as.integer( cols.C[[1]] )
    dims.C[[2]] <- as.integer( cols.C[[2]] - cols.C[[1]] + 1L )
  }

  dims.check.A <- dims.A
  dims.check.B <- dims.B

  if( tp.A ){
    dims.check.A <- rev(dims.A)
  }

  if( tp.B ){
    dims.check.B <- rev(dims.B)
  }

  if( dims.check.A[2] != dims.check.B[1] ||
      dims.check.B[2] != dims.C[2] ||
      dims.check.A[1] != dims.C[1] ){
    stop( "Not all tensors have matching dimensions" )
  }

  # Actual calls
  # L3
  if( tens.A$is.under ){
    ret <- .Call( "cuR_cublas_sgemm",
                  tens.A$ptr,
                  tens.B$ptr,
                  tens.C$ptr,
                  dims.A,
                  dims.B,
                  off.cols.A,
                  off.cols.B,
                  off.cols.C,
                  tp.A,
                  tp.B,
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
    tmp.A <- obj.create( dims.A )
    tmp.B <- obj.create( dims.B )
    tmp.C <- obj.create( dims.C )

    transfer.core( tens.A$ptr, tmp.A, 0L, 0L, "n", dims.A, off.cols.A )
    transfer.core( tens.B$ptr, tmp.B, 0L, 0L, "n", dims.B, off.cols.B )
    transfer.core( tens.C$ptr, tmp.C, 0L, 0L, "n", dims.C, off.cols.C )

    if( tp.A ){
      tmp.A <- t(tmp.A)
    }

    if( tp.B ){
      tmp.B <- t(tmp.B)
    }

    # Math
    tmp.C <- ( alpha * tmp.A ) %*% tmp.B + ( beta * tmp.C )

    transfer.core( tmp.C, tens.C$ptr, 0L, 0L, "n", dims.C, NULL, off.cols.C )
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
