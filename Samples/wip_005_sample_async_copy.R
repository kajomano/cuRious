# This script shows how to use streams and asynchronous tensor push/pull
# operations combined with asychronous kernel launches. This technique is
# utilized to great effect to hide the latency of data transfers between
# the host and the device. This is a longer sample script, so be prepared!
# To keep the console output clean, it is advised to build the package without
# debug prints. To do this, comment out the definition #define DEBUG_PRINTS 1
# in the src/debug.h file.
library( cuRious )
library( microbenchmark )

n <- 1000
mat.dummy <- matrix( rnorm( n*n ), ncol = n )

# Create matrix tensors that will be the I/O points for the operation. Remember,
# it makes sense to create a stage for these tensors.
tens.in <- tensor$new( mat.dummy )
tens.in$create.stage()
tens.in$dive()

tens.out <- tensor$new( mat.dummy )
tens.out$create.stage()
tens.out$dive()

# Create a matrix tensor that will be multiplied with our inputs
tens.trans <- tensor$new( t(diag(1, n, n)) )
tens.trans$dive()

# Create a list of 10 matrices that will form the input data to our system
in.mat.list <- lapply( 1:10, function( ... ){
  matrix( rnorm(n*n), ncol = n )
})

# This list will hold the output of the system
out.mat.list <- list()

# cuBLAS handle
handle <- cublas.handle$new()
handle$activate()

# Synchronous processing ====
# Let's first check how long it takes to do the operations in a synchronous
# fashion. You should already have a rough idea how long this could take
# from the previous sample scripts.

synch.process <- function(){
  out.mat.list <<- lapply( in.mat.list, function( mat ){
    # Transfer the matrix to the GPU memory
    tens.in$push( mat )

    # Do the matrix operation
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )

    # Recover the computed matrix from GPU memory
    tens.out$pull()
  })
}

# Timing of individual function calls
print( microbenchmark( tens.in$push( mat.dummy ), times = 10 ) )
print( microbenchmark( cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 ), times = 10 ) )
print( microbenchmark( tens.out$pull(), times = 10 ) )

# If you add the times from the previous function calls and multiply by
# 10, you should come to the same values that you see here.
print( microbenchmark( synch.process(), times = 10 ) )

# Asynchronous memory transfer ====
# Current Nvidia GPUs are capable of moving data to and from the GPU memory
# simultaneously. To call CUDA functions asynchronously, a separate non-default
# stream needs to be set for each function queue. For this, we need two CUDA
# streams
in.stream <- cuda.stream$new()
in.stream$activate()

out.stream <- cuda.stream$new()
out.stream$activate()

# The push and pull operations not only move the data between the memories, but
# also convert between double and single precision floating point
# representations. This is done on the CPU. This conversion part can be called
# separately for both push and pull if the tensor is staged
print( microbenchmark( tens.in$push.preproc( mat.dummy ), times = 10 ) )
print( microbenchmark( tens.out$pull.proc(), times = 10 ) )

# The actual host<-->device memory transfers can launched separately also by
# calling the .async fetches and supporting a stream on staged tensors
print( microbenchmark( tens.in$push.fetch.async( in.stream ), times = 10 ) )
print( microbenchmark( tens.out$pull.prefetch.async( out.stream ), times = 10 ) )

# These functions return before finishing the actual data transfer, letting us
# do something else in the meantime on the CPU. We can overlay a push and a pull
# operation by pull-prefetching and push-fetching while the CPU converts the data
# going in the other direction like so:
async.transfer <- function(){
  # Prefetch data from GPU and meanwhile preprocess the data to be pushed
  tens.out$pull.prefetch.async( out.stream )
  tens.in$push.preproc( mat.dummy )

  # When finished with preprocessing the pushed data, lunch the asynchronous
  # push-fetch
  tens.in$push.fetch.async( in.stream )

  # Wait for the prefetch to complete, then start processing the pulled data
  cuda.stream.sync( out.stream )
  ret <- tens.out$pull.proc()

  # Wait for the push-fetch to finish and return
  cuda.stream.sync( in.stream )
  ret
}

# Compare the time this takes to the sequential version. As you can see, the
# transfer times are masked behind the CPU conversion
sync.transfer <- function(){
  tens.in$push( mat.dummy )
  tens.out$pull()
}

print( microbenchmark( sync.transfer(), times = 10 ) )
print( microbenchmark( async.transfer(), times = 10 ) )

# ------------------------------------------------------------------------------

# Asynchronous processing ====
# Current CUDA GPUs are also capable of running kernel perations parallel to
# data transfers.

# This means that at This parallel execution scheme is capable of
# hiding the latency overhead from moving data to and from the GPU, removing the
# negative side effects of using the GPU for computation.

trans.stream <- cuda.stream$new()
trans.stream$activate()



# Armed with this knowledge, let's define the async transformation function
asynch.transform <- function(){
  out.mat.list <<- lapply( in.mat.list, function( mat ){
    # Transfer the matrix to the GPU memory
    tens.in$push( mat )

    # Do the matrix operation
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )

    # Return with the result
    tens.out$pull()
  })
}

clean.global()
