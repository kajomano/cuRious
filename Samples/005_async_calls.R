library( cuRious )
library( microbenchmark )

# Devices are capable of executing some functions asynchronously with regards to
# the host thread, given the right circumstances. One such instance is data
# transfers involving L2-L3 tensors.

# L2 tensors are stored in pinned host memory, which is directly accessible by
# the devices. This means that a device can oversee the task of data transfer
# without the need for supervision from the host thread. When invoking an a-
# synchronous transfer, the control is immediately returned to the host thread.

# In order to invoke a data transfer in an asynchronous manner, a stream needs
# be supplied. Streams are action queues, which store, order and execute the
# operations that were issued to a device.

# Let's create a stream:
stream <- cuda.stream$new()

# A synchronous and an asynchronous transfer:
src <- tensor$new( rnorm( 10^6 ), 2L )
dst <- tensor$new( src, 3L, copy = FALSE )

pip.sync  <- pipe$new( src, dst )
pip.async <- pipe$new( src, dst, stream = stream )

print( microbenchmark( pip.sync$run(),  times = 100 ) )
print( microbenchmark( pip.async$run(), times = 100 ) )

# The asynchronous transfer seems much faster, but keep in mind that since the
# control immediately returns to the host thread, the actual transfer might only
# be completed much later. To be able to wait for finished asynchronous calls,
# streams implement the $sync() operation, which does not return until all calls
# are done executing in the streams queue.

# An example of syncing:
pip.synced.run <- function(){
  for( i in 1:100 ){
    pip.async$run()
  }
  stream$sync()
}

print( microbenchmark( pip.synced.run(), times = 10 ) )

clean()
