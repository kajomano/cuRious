library( cuRious )
library( microbenchmark )
library( R6 )

src <- tensor$new( rnorm( 10^6 ) )
dst <- tensor$new( src, init = "mimic" )

tun <- tunnel$new( src, dst )

src$transform( 1L )
