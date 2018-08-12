library( cuRious )
library( microbenchmark )

test <- c( 1, 2 )

tens <- tensor$new( test, 0L, c( 1, 2 ) )

tens$dims
tens$obj
