library( cuRious )
library( microbenchmark )

stream  <- stream$new( )
context <- fusion.context$new( )

stream$destroy()
stream$device <- 1

context$destroy()
context$stream <- stream

context$device <- 1

context$stream <- stream

context$device <- 0

context$deploy()

context$stream <- NULL

context$deploy()
