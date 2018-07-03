require( cuRious )
require( microbenchmark )

# mult <- 1

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )

tens.A.3 <- tensor$new( mat.A, 3 )
tens.B.3 <- tensor$new( mat.A, 3 )

tens.A.0 <- tensor$new( mat.A, 0 )
tens.B.0 <- tensor$new( mat.A, 0 )

unit.A.3 <- tensor$new( 1.0, 3 )
unit.B.3 <- tensor$new( 1.0, 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- thrust.context$new( stream )

unit <- thrust.pow$new( unit.A.3, unit.B.3, context = context )
L3   <- thrust.pow$new( tens.A.3, tens.B.3, subs, subs, context = context )
L0   <- thrust.pow$new( tens.A.0, tens.B.0, subs, subs )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.B.3$pull() )
    print( tens.B.0$pull() )
  }

  identical( tens.B.3$pull(), tens.B.0$pull() )
}
