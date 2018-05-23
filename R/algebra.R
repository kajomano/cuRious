# .Calls: src/algebra.cpp

# B = alpha*A + B
alg.saxpy <- function( tens.A, tens.B, alpha = 1, stream = NULL ){
  # check.tensor.under( tens.A, tens.B )

  if( !is.null( stream ) ){
    check.cuda.stream( stream )
    stream <- stream$stream
  }

  # Results go into tens.B
  ret <- .Call( "cuR_alg_saxpy",
                tens.A$ptr,
                tens.B$ptr,
                tens.A$l,
                alpha,
                stream )

  if( is.null( ret ) ) stop( "Subroutine failed" )

  # TODO ====
  # Have this blocking on the inside

  # # If no stream is given, make this call blocking
  # if( is.null( stream ) ){
  #   cuda.stream.sync.all()
  # }

  invisible( NULL )
}
