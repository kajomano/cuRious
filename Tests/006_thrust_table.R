library( cuRious )
library( microbenchmark )

subs <- c( 1 * mult * mult + 1 , 7 * mult * mult )
vect.x <- as.integer( rep( c( 1, 2, 3, 4 ), times = 2 * mult * mult ) )

tens.x.0 <- tensor$new( vect.x, 0 )
tens.p.0 <- tensor$new( vect.x, 0, copy = FALSE )
tens.w.0 <- tensor$new( NULL, 0, c( 1, 4 ), "i" )
tens.s.0 <- tensor$new( NULL, 0, c( 1, subs[[2]] - subs[[1]] + 1 ), "i" )

tens.x.3 <- tensor$new( tens.x.0, 3 )
tens.p.3 <- tensor$new( tens.p.0, 3 )
tens.w.3 <- tensor$new( tens.w.0, 3 )
tens.s.3 <- tensor$new( tens.s.0, 3 )

unit.x.3 <- tensor$new( 1L, 3 )
unit.p.3 <- tensor$new( 1L, 3 )
unit.w.3 <- tensor$new( 1L, 3 )
unit.s.3 <- tensor$new( 1L, 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- thrust.context$new( stream )

unit <- thrust.table$new( unit.x.3, unit.p.3, unit.w.3, unit.s.3, context = context )
L0   <- thrust.table$new( tens.x.0, tens.p.0, tens.w.0, tens.s.0, subs, subs, context = context )
L3   <- thrust.table$new( tens.x.3, tens.p.3, tens.w.3, tens.s.3, subs, subs, context = context )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.p.3$pull() )
    print( tens.p.0$pull() )

    print( tens.w.3$pull() )
    print( tens.w.0$pull() )

    print( tens.s.3$pull() )
    print( tens.s.0$pull() )
  }

  all(
    test.thr.equality( tens.p.3$pull(), tens.p.0$pull() ),
    test.thr.equality( tens.w.3$pull(), tens.w.0$pull() ),
    test.thr.equality( tens.s.3$pull(), tens.s.0$pull() )
  )
}
