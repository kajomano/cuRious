# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is writte so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas calls (through the whole session) is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cublas.handle",
  public = list(
    create = function(){
      private$handle <- .Call( "cuR_create_cublas_handle" )
      # Check for errors
      if( is.null( private$handle ) ) stop( "The cuBLAS handle could not be created" )
    },
    destroy = function(){
      ret <- .Call( "cuR_destroy_cublas_handle", private$handle )
      # Check for errors
      if( is.null( ret ) ) stop( "The cuBLAS handle could not be destroyed" )
      private$handle <- NULL
    }
  ),
  private = list(
    handle = NULL
  ),
  active = list(
    get.handle = function( val ){
      if( missing(val) ){
        if( self$is.created ) stop( "The cuBLAS handle is not yet created" )
        private$handle
      }
    },
    is.created = function(){
      is.null( private$handle )
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

# TODO ====
# Add the scaling function from cuBLAS!

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

  # TODO ====
  # Rewrite these checks in case of other op choices
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
# C = alpha*op.a(A)*op.b(B) + beta*C
# op.x:
# 'N' - nothing
# 'T' - transpose
# 'H' - conjugate transpose
cublas.sgemm <- function( tens.A, tens.B, tens.C, alpha, beta, op.a = c( 'N', 'T', 'H' ), op.b = op.a, handle ){
  op.choices = c( 'N', 'T', 'H' )
  op.a <- match.arg( op.a, op.choices )
  op.b <- match.arg( op.b, op.choices )
  op.a.int <- which( op.choices == op.a )
  op.b.int <- which( op.choices == op.b )

  # Sanity checks
  if( !all( is.under( tens.A, tens.B, tens.C ) ) ){
    stop( "Not all tensors are under" )
  }

  if( length( tens.A$get.dims ) != 2 || length( tens.B$get.dims ) != 2 || length( tens.C$get.dims ) != 2 ){
    stop( "Not all tensor have the correct number of dimensions" )
  }

  # TODO ====
  # Rewrite these checks in case of other op choices
  # UGHH, this is brain wrecking
  if( tens.A$get.dims[2] != tens.B$get.dims[1] ||
      tens.B$get.dims[2] != tens.C$get.dims[2] ||
      tens.A$get.dims[1] != tens.C$get.dims[1] ){
    stop( "Not all tensor have matching dimensions" )
  }

  # Results go into tens.y
  ret <- .Call( "cuR_cublas_sgemm",
                tens.A$get.tensor,
                tens.B$get.tensor,
                tens.C$get.tensor,
                tens.A$get.dims,
                tens.B$get.dims,
                alpha,
                beta,
                op.a.int,
                op.b.int,
                handle$get.handle )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  invisible( NULL )
}


