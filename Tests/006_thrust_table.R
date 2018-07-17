library( cuRious )
library( microbenchmark )

cols <- 10 * mult + 1
rows <- 10 * mult
subs <- c( 1, 10 * mult )

vect.x <- as.integer( rep( c( 1, 2, 3, 4 ), length.out = cols * mult * mult ) )

tens.x.0 <- cuRious::tensor$new( vect.x, 0 )
tens.p.0 <- cuRious::tensor$new( vect.x, 0, copy = FALSE )
tens.w.0 <- cuRious::tensor$new( NULL, 0, c( 1, 4 ), "i" )
tens.s.0 <- cuRious::tensor$new( NULL, 0, c( 1, subs[[2]] - subs[[1]] + 1 ), "i" )

tens.x.3 <- cuRious::tensor$new( tens.x.0, 3 )
tens.p.3 <- cuRious::tensor$new( tens.p.0, 3 )
tens.w.3 <- cuRious::tensor$new( tens.w.0, 3 )
tens.s.3 <- cuRious::tensor$new( tens.s.0, 3 )

# Mandatory variables
stream   <- cuRious::stream$new( )
context  <- cuRious::thrust.context$new( stream )

L0       <- cuRious::thrust.table$new( tens.x.0, tens.p.0, tens.w.0, tens.s.0, subs, subs )
L3       <- cuRious::thrust.table$new( tens.x.3, tens.p.3, tens.w.3, tens.s.3, subs, subs, context = context )

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
