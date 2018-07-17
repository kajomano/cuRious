require( cuRious )
require( microbenchmark )

# mult <- 100

cols <- 10 * mult + 1
rows <- 10 * mult
subs <- c( 1, 10 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.integer( 1:(cols) )

tens.A.3 <- cuRious::tensor$new( mat.A, 3 )
tens.x.3 <- cuRious::tensor$new( vect.x, 3 )

tens.A.0 <- cuRious::tensor$new( mat.A, 0 )
tens.x.0 <- cuRious::tensor$new( vect.x, 0 )

# Mandatory variables
stream   <- cuRious::stream$new( )
context  <- cuRious::thrust.context$new( stream )

L3       <- cuRious::thrust.cmin.pos$new( tens.A.3, tens.x.3, subs, subs, context = context )
L0       <- cuRious::thrust.cmin.pos$new( tens.A.0, tens.x.0, subs, subs )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.x.3$pull() )
    print( tens.x.0$pull() )
  }

  test.thr.equality( tens.x.3$pull(), tens.x.0$pull() )
}
