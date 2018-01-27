# This script shows how to use streams and asynchronous transfer() calls
# combined with asychronous kernel launches, also known as data streaming. This
# technique is utilized to great effect to hide the latency of data transfers
# between the host and the device.

# To keep the console output clean, it is advised to build the package without
# debug prints. To do this, comment out the definition #define DEBUG_PRINTS 1
# in the src/common.h header file.
library( cuRious )
library( microbenchmark )

# The main goal with streaming is to reduce memory consumption when handling
# large data sets, but also to hide the transfer latency behind the computation.
# For this, the large data needs to be split into smaller parts, which in turn
# requires us to be able to subset the data. R's matrix subsetting is slow
# because it allocates new space for the subset, and copies the data. Since
# we will be transferring this information anyway, the subsetting step can be
# incorporated into the transfer() function. To this extent, all transfers
# between hsot memory locations (level 0,1 and 2) are subsettable.

# There is a catch however: the way R and also CUDA expects data is in
# column-major format, in a continous memory space. Here is an example how this
# looks:
#
# R matrix:     | 1 3 5 |
#               | 2 4 6 |
#
# Memory image: | 1 2 3 4 5 6 |
#
# This makes subsetting rows very difficult, and slow. For this reason, cuRious
# only implements column-wise subsetting. In column-major storage, the rows
# should be the dimensions, and the columns the different observations. This
# probably goes against modern conventions, but hey, we are here for speed, not
# for convenience :D

# Base matrix size
n <- 1000

# Let's create a larger matrix that will serve as our input data, and another
# one that will store the output:
mat.in  <- matrix( round(rnorm( 10*n*n )), nrow = n, ncol = 10*n )
mat.out.R <- matrix( 0, nrow = n, ncol = 10*n )

# This matrix will serve as our processing matrix
mat.proc <- t(diag( 1, n, n ))

# Synchronous processing - R ===================================================
# Let's first try to do what we want to achieve in pure R. We want to multiply
# each n*n sub matrix of mat.in by mat.proc, and store the output in mat.out's
# corresponding columns:
proc.R.sync <- function(){
  for( i in 1:10 ){
    col.subset <- (1+(i-1)*n):(i*n)           # Create a column subset
    mat.subs.in <- mat.in[, col.subset ]      # subset the input matrix
    mat.subs.out <- mat.subs.in %*% mat.proc  # Process the subsetted matrix
    mat.out.R[, col.subset ] <<- mat.subs.out # Store the processed matrix
  }
}

# Run the processing to see how much time this takes
print( microbenchmark( proc.R.sync(), times = 10 ) )

# Synchronous processing - CUDA ================================================
# The same thing can be done with our CUDA tools introduced in the previous
# scripts:

# Create the necessary tensors
mat.dummy <- matrix( 0, n, n )
mat.out.CUDA <- copy( mat.out.R )

tens.in.1 <- tensor$new( mat.dummy )
tens.in.1$dive()

tens.proc <- tensor$new( mat.proc )
tens.proc$dive()

tens.out.1 <- tensor$new( mat.dummy )
tens.out.1$dive()

# cuBLAS handle
handle <- cublas.handle$new()
handle$activate()

proc.CUDA.sync <- function(){
  for( i in 1:10 ){
    # Create a column subset
    col.subset <- (1+(i-1)*n):(i*n)

    # Subset the input matrix and move to the device memory
    transfer( mat.in, tens.in.1, cols.src = col.subset )

    # Process the subsetted matrix
    cublas.sgemm( handle, tens.in.1, tens.proc, tens.out.1, alpha = 1, beta = 0 )

    # Recover the results from device memory and store the processed matrix
    transfer( tens.out.1, mat.out.CUDA, cols.dst = col.subset )
  }
}

# Run the processing to see how much time this takes, and compare it to the
# native R implementation
print( microbenchmark( proc.R.sync(),    times = 10 ) )
print( microbenchmark( proc.CUDA.sync(), times = 10 ) )

# Did we get the same result?
print( identical( mat.out.R, mat.out.CUDA ) )

# This is already faster, but we can make the processing even faster with
# asynchronous parallelization

# Asynchronous processing - CUDA ===============================================
# With asynchronous memory transfer calls (remember, level 2 - 3 transfer calls
# can be asynchronous with regards to the host if a stream is supplied), we can
# parallelize processing and memory transfers:

# Create the necessary tensors
mat.out.CUDA.async <- copy( mat.out.R )

tens.in.stage <- tensor$new( mat.dummy )
tens.in.stage$transform( 2 )

tens.out.stage <- tensor$new( mat.dummy )
tens.out.stage$transform( 2 )

tens.in.2 <- tensor$new( mat.dummy )
tens.in.2$dive()

tens.out.2 <- tensor$new( mat.dummy )
tens.out.2$dive()

# CUDA stream handles. We need 3 of them, one for transferring data in, one for
# processing data, and one for moving data out
stream.in <- cuda.stream$new()
stream.in$activate()

stream.proc <- cuda.stream$new()
stream.proc$activate()

stream.out <- cuda.stream$new()
stream.out$activate()

# Armed with the above knowledge, let's define the async function. We are going
# use a double-buffered approach, hence the tens.in/out.2 definitions above:
proc.CUDA.async <- function(){
  # Spool up
  # Iteration 1
  i <- 1

  # Create a column subset
  col.subset.in <- (1+(i-1)*n):(i*n)

  # Subset the input matrix and move to the stage (level 2 tensor), and then to
  # the dveice memory. We could do this in one step, but later this will be two,
  # so let's keep this also complicated!
  transfer( mat.in, tens.in.stage, cols.src = col.subset.in )

  # Async call, even though we don't need it just yet!
  transfer( tens.in.stage, tens.in.1, stream = stream.in )

  # Wait for everything to finish
  cuda.stream.sync.all()

  # Iteration 2
  i <- 2
  col.subset.in <- (1+(i-1)*n):(i*n)

  # Start processing tens.in.1 right away
  cublas.sgemm( handle, tens.in.1, tens.proc, tens.out.1, 1, 0, stream = stream.proc )

  # Move new data to the device
  transfer( mat.in, tens.in.stage, cols.src = col.subset.in )
  transfer( tens.in.stage, tens.in.2, stream = stream.in )
  cuda.stream.sync.all()

  # Iterations 3-10
  for( i in 3:10 ){
    # Tick or tock (double buffering)
    if( i %% 2 ){
      tens.in       <- tens.in.1
      tens.out      <- tens.out.1
      tens.in.proc  <- tens.in.2
      tens.out.proc <- tens.out.2
    }else{
      tens.in       <- tens.in.2
      tens.out      <- tens.out.2
      tens.in.proc  <- tens.in.1
      tens.out.proc <- tens.out.1
    }

    col.subset.in  <- (1+(i-1)*n):(i*n)
    col.subset.out <- (1+(i-3)*n):((i-2)*n)

    # Start processing
    cublas.sgemm( handle, tens.in.proc, tens.proc, tens.out.proc, 1, 0, stream = stream.proc )

    # Start moving data out (async part), and in (sync part)
    transfer( tens.out, tens.out.stage, stream = stream.out )
    transfer( mat.in, tens.in.stage, cols.src = col.subset.in )

    # Start moving data in (async part), and out (sync part)
    transfer( tens.in.stage, tens.in, stream = stream.in )
    cuda.stream.sync( stream.out )
    transfer( tens.out.stage, mat.out.CUDA.async, cols.dst = col.subset.out )

    cuda.stream.sync.all()
  }

  # Spool down
  # Iteration 11
  i <- 11
  col.subset.out <- (1+(i-3)*n):((i-2)*n)
  cublas.sgemm( handle, tens.in.2, tens.proc, tens.out.2, 1, 0, stream = stream.proc )
  transfer( tens.out.1, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, mat.out.CUDA.async, cols.dst = col.subset.out )

  cuda.stream.sync.all()

  # Iteration 12
  i <- 12
  col.subset.out <- (1+(i-3)*n):((i-2)*n)
  transfer( tens.out.2, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, mat.out.CUDA.async, cols.dst = col.subset.out )

  cuda.stream.sync.all()
}

# Run the processing to see how much time this takes, and compare it to the
# synchronous CUDA implementation, and the native R implementation
print( microbenchmark( proc.R.sync(),     times = 10 ) )
print( microbenchmark( proc.CUDA.async(), times = 10 ) )
print( microbenchmark( proc.CUDA.sync(),  times = 10 ) )

# The async process runtime is governed by the host memory speed in this case,
# hence (only) the 10x speedup compared to the native R implementation. However,
# with more complex processing, or more processing steps, the memory transfer
# times can be completely hidden behind the processing. This would lead to the
# same speedup that we have seen when comparing just the GEMM implementations.

# Did we get the same result?
print( identical( mat.out.CUDA, mat.out.CUDA.async ) )

clean.global()
