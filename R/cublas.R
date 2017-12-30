# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is writte so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas call, or even thorugh the whole session is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cublas.handle",
  public = list(
    initialize = function(){
      private$handle <- .Call( "cuR_create_cublas_handle" )
      # Check for errors
      if( is.null( private$handle ) ) stop( "cuBLAS handle could not be created" )
    }
  ),
  private = list(
    handle = NULL
  ),
  active = list(
    get.handle = function( val ){
      if( missing(val) ) return( private$handle )
    }
  )
)

# cuBLAS linear algebra operations ====

# y = alpha*x + y
# The trick here is that element-wise addition can be this way done also on
# matrices, even though thats not the intended use
cublas.saxpy <- function( tens.x, tens.y, alpha, handle ){
  # Sanity checks
  if( !all( is.under( tens.x, tens.y ) ) ){
    stop( "Not all tensors are under" )
  }

  if( !identical( tens.x$get.l, tens.y$get.l ) ){
    stop( "Not all tensor lengths match" )
  }

  # Results go into tens.y
  ret <- .Call( "cuR_cublas_saxpy",
                tens.x$get.tensor,
                tens.y$get.tensor,
                tens.x$get.l,
                alpha,
                handle$get.handle )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  invisible( NULL )
}

# y = alpha*op(A)*x + beta*y
# op:
# 'N' - nothing
# 'T' - transpose
# 'H' - conjugate transpose
cublas.sgemv <- function( tens.A, tens.x, tens.y, alpha, beta, op = c( 'N', 'T', 'H' ), handle ){
  op.choices = c( 'N', 'T', 'H' )
  op <- match.arg( op, op.choices )
  op.int <- which( op.choices == op )

  # Sanity checks
  if( !all( is.under( tens.A, tens.x, tens.y ) ) ){
    stop( "Not all tensors are under" )
  }

  if( length( tens.A$get.dims ) != 2 || length( tens.x$get.dims ) != 1 || length( tens.y$get.dims ) != 1 ){
    stop( "Not all tensor have the correct number of dimensions" )
  }

  if( tens.A$get.dims[2] != tens.x$get.dims || tens.A$get.dims[1] != tens.y$get.dims ){
    stop( "Not all tensor have matching dimensions" )
  }

  # Results go into tens.y
  ret <- .Call( "cuR_cublas_sgemv",
                tens.A$get.tensor,
                tens.x$get.tensor,
                tens.y$get.tensor,
                tens.A$get.dims,
                alpha,
                beta,
                op.int,
                handle$get.handle )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  invisible( NULL )
}


