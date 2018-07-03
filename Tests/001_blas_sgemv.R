require( cuRious )
require( microbenchmark )

# gemv
# y.span(y) = alpha*A.tp(A.span(A)) %*% x.span(x) + beta*y.span(y)
# mult <- 1

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols) )
vect.y <- as.double( 1:(cols) )

tens.A.3 <- tensor$new( mat.A  , 3 )
tens.x.3 <- tensor$new( vect.x , 3 )
tens.y.3 <- tensor$new( vect.y , 3 )

tens.A.0 <- tensor$new( mat.A  , 0 )
tens.x.0 <- tensor$new( vect.x , 0 )
tens.y.0 <- tensor$new( vect.y , 0 )

unit.A.3 <- tensor$new( 1.0, 3 )
unit.x.3 <- tensor$new( 1.0, 3 )
unit.y.3 <- tensor$new( 1.0, 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- cublas.context$new( stream )

unit <- cublas.sgemv$new( unit.A.3, unit.x.3, unit.y.3, context = context )
L3   <- cublas.sgemv$new( tens.A.3, tens.x.3, tens.y.3, subs, subs, subs, TRUE, context = context )
L0   <- cublas.sgemv$new( tens.A.0, tens.x.0, tens.y.0, subs, subs, subs, TRUE )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.y.3$pull() )
    print( tens.y.0$pull() )
  }

  identical( tens.y.3$pull(), tens.y.0$pull() )
}
