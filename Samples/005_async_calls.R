library( cuRious )
library( microbenchmark )

# Devices are capable of executing some functions asynchronously with regards to
# the host thread, given the right circumstances. cuRious supports asynchronous
# execution on all fusion calls, including all pipe calls.

# In order to invoke a pipe transfer in an asynchronous manner, an active stream
# needs be supplied to the pipe context. Streams are action queues, which store,
# order and execute the operations that were issued to a device.

# Let's create a stream. Streams are also created undeployed by default, if no
# deployment target is set:
stream <- stream$new()$deploy( 1 )

# Pipe contexts
pip.cont.sync  <- pipe.context$new()$deploy( 1 )
pip.cont.async <- pipe.context$new( stream = stream )$deploy( 1 )

# A synchronous and an asynchronous transfer:
src <- tensor$new( rnorm( 10^6 ) )
dst <- tensor$new( src, copy = FALSE )

pip.sync  <- pipe$new( src, dst, context = pip.cont.sync )
pip.async <- pipe$new( src, dst, context = pip.cont.async )

print( microbenchmark( pip.sync$run(),  times = 100 ) )
print( microbenchmark( pip.async$run(), times = 100 ) )

# The asynchronous transfer seems much faster, but keep in mind that since the
# control immediately returns to the host thread, the actual transfer might only
# be completed much later. To be able to wait for finished asynchronous calls,
# streams implement the $sync() operation, which does not return until all calls
# are done executing in the streams queue.

# An example of syncing:
pip.synced.run <- function(){
  pip.async$run()
  stream$sync()
}

print( microbenchmark( pip.synced.run(), times = 100 ) )

clean()
