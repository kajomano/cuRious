library( cuRious )
library( microbenchmark )
library( R6 )

src <- tensor$new( rnorm( 10^3 ) )
dst <- tensor$new( src, init = "mimic" )

tun <- tunnel$new( src, dst )

microbenchmark( tun$transfer(), times = 100 )
microbenchmark( transfer( src, dst ), times = 100 )

src$destroy()
