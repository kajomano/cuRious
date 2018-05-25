library( cuRious )
library( microbenchmark )

src <- tensor$new( 1:10 )
dst <- tensor$new( 1:10 )
pip <- pipe$new( src, dst )

microbenchmark( pip$run() )

var1 <- 4:13
src$ptr <- var1
var2 <- src$ptr

src$push( 1:10 )

var1
var2
