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
mat.dummy <- matrix( 0, ncol = n, nrow = n )

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

# Create a list of 10 matrices that will form the output palceholders to our
# system. Watch out, we cannot use mat.dummy for this purpose as it would be
# soft-copied, causing every one of them to be overwritten simultaneously when
# using one in a pull() function!
out.mat.list <- lapply( 1:10, function( ... ){
  matrix( 0, ncol = n, nrow = n )
})

# cuBLAS handle
handle <- cublas.handle$new()
handle$activate()

# Synchronous processing ====
# Let's first check how long it takes to do the operations in a synchronous
# fashion. You should already have a rough idea how long this could take
# from the previous sample scripts.

sync.process <- function(){
  for( i in 1:10 ){
    # Transfer the matrix to the GPU memory
    tens.in$push( in.mat.list[[i]] )

    # Do the matrix operation
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )

    # Recover the computed matrix from GPU memory
    tens.out$pull( out.mat.list[[i]] )
  }
}

# Timing of individual function calls
print( microbenchmark( tens.in$push( mat.dummy ), times = 10 ) )
print( microbenchmark( cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 ), times = 10 ) )
print( microbenchmark( tens.out$pull( mat.dummy ), times = 10 ) )

# If you add the times from the previous function calls and multiply by
# 10, you should come to the same values that you see here.
print( microbenchmark( sync.process(), times = 10 ) )

# Asynchronous memory transfer ====
# Some Nvidia GPUs are capable of moving data to and from the GPU memory
# simultaneously. To call CUDA functions asynchronously, a separate non-default
# stream needs to be set for each function queue. For simultaneous push and pull
# we need two CUDA streams
in.stream <- cuda.stream$new()
in.stream$activate()

out.stream <- cuda.stream$new()
out.stream$activate()

# The push and pull operations not only move the data between the memories, but
# also convert between double and single precision floating point
# representations. This is done on the CPU. This conversion part can be called
# separately for both push and pull if the tensor is staged
print( microbenchmark( tens.in$push.preproc( mat.dummy ), times = 10 ) )
print( microbenchmark( tens.out$pull.proc( mat.dummy ), times = 10 ) )

# The actual host<-->device memory transfers can be launched separately also by
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
  tens.out$pull.proc( mat.dummy )

  # Wait for the push-fetch to finish and return
  cuda.stream.sync( in.stream )
}

# Compare the time this takes to the sequential version. As you can see, the
# transfer times are masked behind the CPU conversion.
sync.transfer <- function(){
  tens.in$push( mat.dummy )
  tens.out$pull( mat.dummy )
}
print( microbenchmark( sync.transfer(), times = 10 ) )
print( microbenchmark( async.transfer(), times = 10 ) )

# While the gain is not much as most of the time is spent processing the data in
# the CPU anyway, async memory copies also allow us to do another thing:

# Asynchronous processing ====
# Current CUDA GPUs are also able to run kernel operations parallel to
# data transfers. This parallel execution scheme is capable of entirely hiding
# the latency overhead from moving data to and from the GPU, removing the
# negative side effects of using the GPU for computation.

# For the parallel cuBLAS kernel we need another stream
cublas.stream <- cuda.stream$new()
cublas.stream$activate()

# At any time, there will be always one push, one pull and one gemm operation.
# The gemm operation requires separate memory as input and output areas (can not
# be done in-place). This requires 4 tensors all together. These 4 tensors will
# all be at some point input and output areas as well. For this, we will define
# 4 tensors, all staged
tens.list <- lapply( 1:4, function(...){
  tens <- tensor$new( mat.dummy )
  tens$create.stage()
  tens$dive()
  tens
})

# To make this easier, here is the rotating scheme. The roles of the tesors will
# be shifted up with each iteration, or the tensors shifted down.
# tens1 : push( new.matrix )
# tens2 : input to gemm
# tens3 : output of gemm
# tens4 : pull( processed.matrix )

# Let's keep the output separate from the synchonous case
out.mat.list.async <- lapply( 1:10, function( ... ){
  matrix( 0, ncol = n, nrow = n )
})

# Armed with the above knowledge, let's define the async function
async.process <- function(){
  # Spool up
  # Iteration 1
  tens.list[[1]]$push.preproc( in.mat.list[[1]] )
  tens.list[[1]]$push.fetch.async( in.stream )
  cuda.stream.sync.all()

  # Iteration 2
  cublas.sgemm( handle, tens.list[[1]], tens.trans, tens.list[[2]], 1, 0, stream = cublas.stream  )
  tens.list[[4]]$push.preproc( in.mat.list[[2]] )
  tens.list[[4]]$push.fetch.async( in.stream )
  cuda.stream.sync.all()

  # Iterations 3-10
  for( i in 3:10 ){
    tens.push     <- tens.list[[4 - ((i+2) %% 4)]]
    tens.gemm.in  <- tens.list[[4 - ((i+1) %% 4)]]
    tens.gemm.out <- tens.list[[4 - ((i)   %% 4)]]
    tens.pull     <- tens.list[[4 - ((i+3) %% 4)]]

    cublas.sgemm( handle, tens.gemm.in, tens.trans, tens.gemm.out, 1, 0, stream = cublas.stream )

    tens.pull$pull.prefetch.async( out.stream )
    tens.push$push.preproc( in.mat.list[[i]] )
    tens.push$push.fetch.async( in.stream )
    cuda.stream.sync( out.stream )
    tens.pull$pull.proc( out.mat.list.async[[i-2]] )

    cuda.stream.sync.all()
  }

  # Spool down
  # Iteration 11
  cublas.sgemm( handle, tens.list[[4]], tens.trans, tens.list[[1]], 1, 0, stream = cublas.stream )
  tens.list[[2]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  tens.pull$pull.proc( out.mat.list.async[[9]] )

  # Iteration 12
  tens.list[[1]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  tens.list[[1]]$pull.proc( out.mat.list.async[[10]] )
}

# Let's check how much time we gained. For reference the pure gemm operation is
# also included
gemm.dummy <- function(){
  for( i in 1:10 ){
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )
  }
}
print( microbenchmark( sync.process(), times = 10 ) )
print( microbenchmark( async.process(), times = 10 ) )
print( microbenchmark( gemm.dummy(), times = 10 ) )

# As you can see, the data transfer times are made up mostly of CPU processing
# time, while the GEMM operation is completely hidden. Of course in a neural
# network, more than one opearations (GEMM or other) will be done in a single
# pass, which means the roles will swithc, and the data transfer times will be
# hidden behind the computation.

# The two results are indeed the same
print( identical( out.mat.list[[1]], out.mat.list.async[[1]] ) )

clean.global()
