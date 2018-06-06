library( cuRious )
library( microbenchmark )

tens0 <- tensor$new( matrix( as.double( 1:10^6 ), 1000, 1000 ), 0L )
tens3 <- tensor$new( tens0, 3L )

thrust0 <- thrust.pow2$new( tens0, tens0 )
thrust3 <- thrust.pow2$new( tens3, tens3 )

print( microbenchmark( thrust0$run() ) )
print( microbenchmark( thrust3$run() ) )
