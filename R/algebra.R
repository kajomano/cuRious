# .Calls: src/algebra.c

# Element-wise operations of 2 tensors, the result ending up in a 3rd (in-place)
# The resulting tensor can be one of the other 2, or all 3 can be the same
# Currently supported operations:
# +, -, *, /
ewop <- function( tens.l, tens.r, tens.res, op = "+" ){
  op.choices <- c( "+", "-", "*", "/" )
  op <- match.arg( op, op.choices )
  op.int <- which( op.choices == op )

  # Sanity checks
  if( !all( is.under( tens.l, tens.r, tens.res ) ) ){
    stop( "Not all tensors are under" )
  }

  if( !identical( tens.l$get.dims, tens.r$get.dims ) ||
      !identical( tens.l$get.dims, tens.res$get.dims ) ){
    stop( "Not all dimensions match" )
  }

  .Call( "ewop",
         tens.l$tensor,
         tens.r$tensor,
         tens.res$tensor,
         op.int,
         length(tens.l$get.dims),
         tens.l$get.dims )

  invisible( NULL )
}
