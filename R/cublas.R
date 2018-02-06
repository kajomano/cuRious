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
    activate = function(){
      if( self$is.active ){
        warning( "The cuBLAS handle has already been activated" )
        return( invisible( self ) )
      }
      private$handle <- .Call( "cuR_activate_cublas_handle" )

      if( is.null( private$handle ) ){
        stop( "The cuBLAS handle could not be activated" )
      }

      invisible( self )
    },
    deactivate = function(){
      if( !self$is.active ){
        warning( "The cuBLAS handle is not active" )
        return( invisible( self ) )
      }

      .Call( "cuR_deactivate_cublas_handle", private$handle )
      private$handle <- NULL

      invisible( self )
    }
  ),

  private = list(
    handle = NULL
  ),

  active = list(
    get.handle = function( val ){
      if( missing(val) ){
        # This produces an error because wherever you need a cublas handle it is
        # not optional
        if( !self$is.active ){
          stop( "The cuBLAS handle has not yet been activated" )
        }
        private$handle
      }
    },
    is.active = function(){
      !is.null( private$handle )
    }
  )
)

# Helper functions ====
is.cublas.handle <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "cublas.handle" %in% class( obj )
  })
}

check.cublas.handle <- function( ... ){
  if( !all( is.cublas.handle( ... ) ) ){
    stop( "Not all objects are cuBLAS handles" )
  }
}

is.cublas.handle.active <- function( ... ){
  check.cublas.handle( ... )

  handles <- list( ... )
  sapply( handles, function( handle ){
    handle$is.active
  })
}

check.cublas.handle.active <- function( ... ){
  if( !all( is.cublas.handle.active( ... ) ) ){
    stop( "Not all cuBLAS handles are active" )
  }
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
# Add sgeam from cuBLAS, remove saxpy!

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
  check.cublas.handle.active( handle )
  check.tensor.under( tens.A, tens.B, tens.C )
  if( !is.null( stream ) ){
    check.cuda.stream( stream )
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
                tens.A$get.obj,
                tens.B$get.obj,
                tens.C$get.obj,
                tens.A$get.dims,
                tens.B$get.dims,
                alpha,
                beta,
                tp.A,
                tp.B,
                handle$get.handle,
                stream )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  # If no stream is given, make this call blocking
  if( is.null( stream ) ){
    cuda.stream.sync.all()
  }

  invisible( TRUE )
}

# B = alpha*A + B
# The trick here is that element-wise addition can be done this way also on
# matrices, even though thats not the intended use
cublas.saxpy <- function( handle, tens.A, tens.B, alpha = 1, stream = NULL ){
  check.cublas.handle.active( handle )
  check.tensor.under( tens.A, tens.B )
  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$get.stream
  }

  # Results go into tens.B
  ret <- .Call( "cuR_cublas_saxpy",
                tens.A$get.obj,
                tens.B$get.obj,
                tens.A$get.l,
                alpha,
                handle$get.handle,
                stream )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  # If no stream is given, make this call blocking
  if( is.null( stream ) ){
    cuda.stream.sync.all()
  }

  invisible( NULL )
}
