require( cuRious )
require( microbenchmark )

# gemv
# y.span(y) = alpha*A.tp(A.span(A)) %*% x.span(x) + beta*y.span(y)
# mult <- 1

cols <- 10 * mult + 1
rows <- 10 * mult
subs <- c( 1, 10 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols) )
vect.y <- as.double( 1:(cols) )

tens.A.3 <- cuRious::tensor$new( mat.A,  3 )
tens.x.3 <- cuRious::tensor$new( vect.x, 3 )
tens.y.3 <- cuRious::tensor$new( vect.y, 3 )

tens.A.0 <- cuRious::tensor$new( mat.A,  0 )
tens.x.0 <- cuRious::tensor$new( vect.x, 0 )
tens.y.0 <- cuRious::tensor$new( vect.y, 0 )

# Mandatory variables
stream   <- cuRious::stream$new()
context  <- cuRious::cublas.context$new( stream )

L3       <- cublas.sgemv$new( tens.A.3, tens.x.3, tens.y.3, subs, subs, subs, TRUE, context = context )
L0       <- cublas.sgemv$new( tens.A.0, tens.x.0, tens.y.0, subs, subs, subs, TRUE )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.y.3$pull() )
    print( tens.y.0$pull() )
  }

  test.thr.equality( tens.y.3$pull(), tens.y.0$pull() )
}

clear <- function(){
  tens.y.3$push( vect.y )
}
