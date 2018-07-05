require( cuRious )
require( microbenchmark )

# mult <- 100

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.integer( 1:(cols) )

tens.A.3 <- tensor$new( mat.A, 3 )
tens.x.3 <- tensor$new( vect.x, 3 )

tens.A.0 <- tensor$new( mat.A, 0 )
tens.x.0 <- tensor$new( vect.x, 0 )

unit.A.3 <- tensor$new( 1.0, 3 )
unit.x.3 <- tensor$new( as.integer( 1 ), 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- thrust.context$new( stream )

unit <- thrust.cmin.pos$new( unit.A.3, unit.x.3, context = context )
L3   <- thrust.cmin.pos$new( tens.A.3, tens.x.3, subs, subs, context = context )
L0   <- thrust.cmin.pos$new( tens.A.0, tens.x.0, subs, subs )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.x.3$pull() )
    print( tens.x.0$pull() )
  }

  test.thr.equality( tens.x.3$pull(), tens.x.0$pull() )
}
