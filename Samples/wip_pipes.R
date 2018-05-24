library( cuRious )
library( microbenchmark )
library( R6 )

src <- tensor$new( rnorm( 10^6 ) )
dst <- tensor$new( src, init = "mimic" )

pip <- pipe$new( src, dst )

print( microbenchmark( pip$run(), times = 100 ) )
print( microbenchmark( transfer( src, dst ), times = 100 ) )

src$destroy()
pip$destroy()
dst$destroy()
