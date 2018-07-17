require( cuRious )
require( microbenchmark )

cols <- 10 * mult + 1
rows <- 10 * mult
subs <- c( 1, 10 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )

tens.A.3 <- cuRious::tensor$new( mat.A, 3 )
tens.B.3 <- cuRious::tensor$new( mat.A, 3 )

tens.A.0 <- cuRious::tensor$new( mat.A, 0 )
tens.B.0 <- cuRious::tensor$new( mat.A, 0 )

# Mandatory variables
stream   <- cuRious::stream$new()
context  <- cuRious::thrust.context$new( stream )

L3       <- cuRious::thrust.pow$new( tens.A.3, tens.B.3, subs, subs, context = context )
L0       <- cuRious::thrust.pow$new( tens.A.0, tens.B.0, subs, subs )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.B.3$pull() )
    print( tens.B.0$pull() )
  }

  test.thr.equality( tens.B.3$pull(), tens.B.0$pull() )
}
