library( cuRious )
library( microbenchmark )

stream1 <- stream$new()
cublas.context <- cublas.context$new(  )


# src <- tensor$new( rnorm( 10^6 ), 2L )
# dst <- tensor$new( src, 3L, copy = FALSE )
#
# # pip.sync  <- pipe$new( src, dst )
# pip.async <- pipe$new( src, dst, stream = stream )
#
# # print( microbenchmark( pip.sync$run(),  times = 100 ) )
# # print( microbenchmark( pip.async$run(), times = 100 ) )
#
# for( i in 1:10 ){
#   pip.async$run()
# }
# stream$sync()
#
# clean()

# context <- fusion.context$new( )
#
# stream$destroy()
# stream$device <- 1
#
# context$destroy()
# context$stream <- stream
#
# context$device <- 1
#
# context$stream <- stream
#
# context$device <- 0
#
# context$deploy()
#
# context$stream <- NULL
#
# context$deploy()
