library( cuRious )
library( microbenchmark )

src <- tensor$new( )
dst <- tensor$new( )
pip <- pipe$new( src, dst )
# pip$run()

microbenchmark( pip$run(), times = 10 )

microbenchmark( src$sever.refs(), times = 100 )

# test.list <- list()
#
# test.list[ c("a", "b") ] <- list( 1, 2 )

# pip$run()
# dst$clear()
# src$ptr <- 4:13
# pip$run()
# dst$ptr

#
# var1 <- 4:13
# src$ptr <- var1
# var2 <- src$ptr
#
# src$push( 1:10 )
#
# var1
# var2
