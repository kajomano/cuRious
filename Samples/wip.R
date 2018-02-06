library( cuRious )
library( microbenchmark )

n <- 10L
tens.x <- tensor$new( matrix( 1, n, n ) )
tens.y <- tensor$new( matrix( 0, n, n ) )

microbenchmark( transfer(tens.x, tens.y), times = 10 )
microbenchmark( transfer.core( tens.x$get.obj,
                               tens.y$get.obj,
                               0L,
                               0L,
                               "n",
                               c( n, n)
                               ), times = 10 )
