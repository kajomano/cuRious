library( cuRious )
library( microbenchmark )

stream1  <- stream$new()
stream1$destroy()

tensor1 <- tensor$new( 1.0 )
tensor2 <- tensor$new()

transfer( tensor1, tensor2, stream = stream1 )

tensor1$pull()
tensor2$pull()


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
