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
    create = function(){
      if( self$is.created ){
        stop( "The cuBLAS handle has already been created" )
      }
      private$handle <- .Call( "cuR_create_cublas_handle" )

      if( is.null( private$handle ) ){
        stop( "The cuBLAS handle could not be created" )
      }

      invisible( NULL )
    },
    destroy = function(){
      if( !self$is.created ){
        stop( "The cuBLAS handle has not yet been created" )
      }

      .Call( "cuR_destroy_cublas_handle", private$handle )
      private$handle <- NULL
    }
  ),

  private = list(
    handle = NULL
  ),

  active = list(
    get.handle = function( val ){
      if( missing(val) ){
        if( !self$is.created ){
          stop( "The cuBLAS handle has not yet been created" )
        }
        private$handle
      }
    },
    is.created = function(){
      !is.null( private$handle )
    }
  )
)

is.cublas.handle <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "cublas.handle" %in% class( obj )
  })
}

is.cublas.handle.created <- function( ... ){
  if( !all( is.cublas.handle( ... ) ) ){
    stop( "Object is not a cublas handle" )
  }

  handles <- list( ... )
  sapply( handles, function( handle ){
    handle$is.created
  })
}

# cuBLAS linear algebra operations ====

# TODO ====
# Add scopy from cuBLAS!

# TODO ====
# Add sswap from cuBLAS!

# TODO ====
# Add sscal from cuBLAS!

# TODO ====
# Add sasum from cuBLAS!

# TODO ====
# Add samin/max from cuBLAS!

# TODO ====
# Add sgeam from cuBLAS!

# TODO ====
# Add sdgmm from cuBLAS!

# TODO BIG ====
# Add reuse to functions when using tensors with fewer dimensions( scalar with
# vectors, scalar and vector with matrices)
# This might be a tradeoff

# C = alpha*tp.a(A)*tp.b(B) + beta*C
# tp = transpose
cublas.sgemm <- function( handle, tens.A, tens.B, tens.C, alpha = 1, beta = 1, tp.A = FALSE, tp.B = FALSE, stream = NULL ){
  # Sanity checks
  if( !is.cublas.handle.created( handle ) ){
    stop( "Cublas handle is not created" )
  }

  if( !all( is.under( tens.A, tens.B, tens.C ) ) ){
    stop( "Not all tensors are under" )
  }

  if( !is.null( stream ) ){
    if( !is.cuda.stream.created( stream ) ){
      stop( "The CUDA stream is not created" )
    }

    stream <- stream$get.stream
  }

  if( tp.A ){
    A.dims <- rev(tens.A$get.dims)
  }else{
    A.dims <- tens.A$get.dims
  }

  if( tp.B ){
    B.dims <- rev(tens.B$get.dims)
  }else{
    B.dims <- tens.B$get.dims
  }

  C.dims <- tens.C$get.dims

  if( A.dims[2] != B.dims[1] ||
      B.dims[2] != C.dims[2] ||
      A.dims[1] != C.dims[1] ){
    stop( "Not all tensor have matching dimensions" )
  }

  # Results go into tens.B
  ret <- .Call( "cuR_cublas_sgemm",
                tens.A$get.tensor,
                tens.B$get.tensor,
                tens.C$get.tensor,
                tens.A$get.dims,
                tens.B$get.dims,
                alpha,
                beta,
                tp.A,
                tp.B,
                handle$get.handle,
                stream )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  invisible( NULL )
}

# TODO ====
# Update saxpy to current state (streams and sanity checks whatnot)
# # B = alpha*A + B
# # The trick here is that element-wise addition can be done this way also on
# # matrices, even though thats not the intended use
# cublas.saxpy <- function( handle, tens.A, tens.B, alpha = 1 ){
#   # Sanity checks
#   if( !is.cublas.handle.created( handle ) ){
#     stop( "Cublas handle is not created" )
#   }
#
#   if( !all( is.under( tens.A, tens.B ) ) ){
#     stop( "Not all tensors are under" )
#   }
#
#   if( !identical( tens.A$get.l, tens.B$get.l ) ){
#     stop( "Not all tensor lengths match" )
#   }
#
#   # Results go into tens.B
#   ret <- .Call( "cuR_cublas_saxpy",
#                 tens.A$get.tensor,
#                 tens.B$get.tensor,
#                 tens.A$get.l,
#                 alpha,
#                 handle$get.handle )
#
#   if( is.null( ret ) ) stop( "Subroutine failed" )
#
#   invisible( NULL )
# }
