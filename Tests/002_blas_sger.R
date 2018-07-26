require( cuRious )
require( microbenchmark )

# A = a*x %*% tp(y) + A
# Create matrix tensors and store them in GPU memory
# mult <- 100

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

L3       <- cuRious::cublas.sger$new( tens.x.3, tens.y.3, tens.A.3, subs, subs, subs, context = context )
L0       <- cuRious::cublas.sger$new( tens.x.0, tens.y.0, tens.A.0, subs, subs, subs )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.A.3$pull() )
    print( tens.A.0$pull() )
  }

  test.thr.equality( tens.A.3$pull(), tens.A.0$pull() )
}

clear <- function(){
  tens.A.3$push( mat.A )
}
