library( cuRious )
library( microbenchmark )

# This script shows how to use streams and asynchronous transfer() calls
# combined with asychronous kernel launches, also known as data streaming. This
# technique is utilized to great effect to hide the latency of data transfers
# between the host and the device.

# To keep the console output clean, it is advised to build the package without
# debug prints. To do this, comment out the definition #define DEBUG_PRINTS 1
# in the src/common.h header file.

# The main goal of streaming is to reduce memory consumption when handling
# large data sets on the device, while also hiding the transfer latency behind
# computation. Large datasets are split into smaller chunks by span subsetting,
# explained in an earlier sample script.

# Base matrix size
n <- 1000

# Let's create a larger matrix that will serve as our input data, and another
# one that will store the output:
mat.in    <- matrix( round(rnorm( 10*n*n )), nrow = n, ncol = 10*n )
mat.out.R <- matrix( 0, nrow = n, ncol = 10*n )

# This matrix will serve as our processing matrix for the gemm operation
mat.proc <- t(diag( 1, n, n ))

# Synchronous processing - R ===================================================
# Let's do the processing in pure R. We want to multiply each n*n sub matrix of
# mat.in by mat.proc, and store the output in mat.out's corresponding columns:
proc.R.sync <- function(){
  for( i in 1:10 ){
    col.subset <- (1+(i-1)*n):(i*n)           # Create a column subset
    mat.subs.in <- mat.in[, col.subset ]      # Subset the input matrix
    mat.subs.out <- mat.subs.in %*% mat.proc  # Process the subsetted matrix
    mat.out.R[, col.subset ] <<- mat.subs.out # Store the processed matrix
  }
}

# Run the processing to see how much time this takes (be patient)
bench.R.sync <- microbenchmark( proc.R.sync(), times = 10 )
print( bench.R.sync )

# Synchronous processing - CUDA ================================================
# The same thing can be done with our CUDA tools introduced in the previous
# scripts:

# Create the necessary tensors
tens.in       <- tensor$new( mat.in, 0L )
tens.out.CUDA <- tensor$new( tens.in, copy = FALSE )

tens.in.1  <- tensor$new( NULL, 3L, c( n, n ) )
tens.proc  <- tensor$new( mat.proc, 3L )
tens.out.1 <- tensor$new( NULL, 3L, c( n, n ) )

# cuBLAS handle
handle <- cublas.handle$new()

# Create transfer pipes
pipes.in <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.in, tens.in.1, src.span = span )
})

pipes.out <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.out.1, tens.out.CUDA, dst.span = span )
})

# Create the gemm operation
gemm <- cublas.sgemm$new( tens.in.1,
                          tens.proc,
                          tens.out.1,
                          alpha = 1,
                          beta = 0,
                          handle = handle )

proc.CUDA.sync <- function(){
  for( i in 1:10 ){
    # Subset the input matrix and move to the device memory
    pipes.in[[i]]$run()

    # Process the subsetted matrix
    gemm$run()

    # Recover the results from device memory and store the processed matrix
    pipes.out[[i]]$run()
  }
}

# Run the processing to see how much time this takes, and compare it to the
# native R implementation
bench.CUDA.sync <- microbenchmark( proc.CUDA.sync(), times = 10 )
print( bench.R.sync )
print( bench.CUDA.sync )

# Did we get the same result?
print( identical( mat.out.R, tens.out.CUDA$ptr ) )

# This is already faster, but we can make the processing even faster with
# asynchronous parallelization

# Asynchronous processing - CUDA ===============================================
# With asynchronous memory transfer calls (remember, level 2 - 3 transfer calls
# can be asynchronous with regards to the host if a stream is supplied), we can
# parallelize processing and memory transfers by double buffering incoming and
# outgoing pipes:

# Time --->
#
# L0       [tens.in]                       [tens.out.CUDA.async]
# L0                \                     /
# -------------------\-------------------/--------------------------------------
# L2                  \                 /
# L2                   [tens.in.stage] /
# L2                                  X
# L2                  [tens.out.stage] \
# L2                 /                  \
# ==================/====================\======================================
# L3               /                      \
# L3   [tens.out.1]                        [tens.in.1]
# L3
# L3   [tens.in.2] ---------> GEMM -----------> [tens.out.2]

# Every other iteration, tens.in.x-s and outs need to be flipped.

# Let's do this:
# Create another output tensor
tens.out.CUDA.async <- tensor$new( tens.out.CUDA, copy = FALSE )

# Asynchronous transfers require L2 staging buffers:
tens.in.stage  <- tensor$new( NULL, 2L, c( n, n ) )
tens.out.stage <- tensor$new( NULL, 2L, c( n, n ) )

# Since we are going to use double buffering, we need additional arrival/depar-
# ture tensors:
tens.in.2  <- tensor$new( NULL, 3L, c( n, n ) )
tens.out.2 <- tensor$new( NULL, 3L, c( n, n ) )

# CUDA stream handles. We need 3 of them, one for transferring data in, one for
# processing data, and one for moving data out
stream.in   <- cuda.stream$new()
stream.proc <- cuda.stream$new()
stream.out  <- cuda.stream$new()

# The most important modification compared to the synchronous operation to
# achieve double buffering is the many new pipes:
pipes.in.L0.L2 <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.in, tens.in.stage, src.span = span )
})

pipes.out.L2.L0 <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.out.stage, tens.out.CUDA.async, dst.span = span )
})

pipes.in.L2.L3 <- list(
  pipe$new( tens.in.stage, tens.in.1, stream = stream.in ),
  pipe$new( tens.in.stage, tens.in.2, stream = stream.in )
)

pipes.out.L3.L2 <- list(
  pipe$new( tens.out.1, tens.out.stage, stream = stream.out ),
  pipe$new( tens.out.2, tens.out.stage, stream = stream.out )
)

# And finally the async gemm operations. Watch out, the tens.in.x and out
# order is flipped compared to the buffering pipes!
async.gemms <- list(
  cublas.sgemm$new( tens.in.2,
                    tens.proc,
                    tens.out.2,
                    alpha = 1,
                    beta = 0,
                    handle = handle,
                    stream = stream.proc ),
  cublas.sgemm$new( tens.in.1,
                    tens.proc,
                    tens.out.1,
                    alpha = 1,
                    beta = 0,
                    handle = handle,
                    stream = stream.proc )
)

# Armed with the above knowledge, let's define the async function:
proc.CUDA.async <- function(){
  # Spool up
  # Iteration 1
  i <- 1

  # Subset the input matrix and move to the stage (level 2 tensor), and then to
  # the device memory.
  pipes.in.L0.L2[[i]]$run()                 # Sync
  pipes.in.L2.L3[[i %% 2 + 1]]$run()        # Async: stream.in

  # Wait for everything to finish
  cuda.device.sync()

  # Iteration 2
  i <- 2

  # Start processing the previously filled tens.in
  async.gemms[[i %% 2 + 1]]$run()           # Async: stream.proc

  # Move new data to the device
  pipes.in.L0.L2[[i]]$run()                 # Sync
  pipes.in.L2.L3[[i %% 2 + 1]]$run()        # Async: stream.in

  # Wait for everything to finish
  cuda.device.sync()

  # Iterations 3-10
  for( i in 3:10 ){
    # Start processing
    async.gemms[[i %% 2 + 1]]$run()         # Async: stream.proc-|
    #                                                            |
    # Start moving data out to the host                          |
    pipes.out.L3.L2[[i %% 2 + 1]]$run()     # Async: stream.out  |
    #                                                    |       |
    # Start moving new data to the device                |       |
    pipes.in.L0.L2[[i]]$run()               # Sync       |       |
    #                                                    |       |
    # Wait for the async data out to finish <------------|       |
    stream.out$sync()  #                                         |
    #                                                            |
    # Finish moving new data to the device                       |
    pipes.in.L2.L3[[i %% 2 + 1]]$run()      # Async: stream.in   |
    #                                                    |       |
    # Finish moving data out to the host                 |       |
    pipes.out.L2.L0[[i-2]]$run()            # Sync       |       |
    #                                                    |       |
    # Wait for everything to finish <--------------------|-------|
    cuda.device.sync()
  }

  # Spool down
  # Iteration 11
  i <- 11
  async.gemms[[i %% 2 + 1]]$run()       # Async: stream.proc
  pipes.out.L3.L2[[i %% 2 + 1]]$run()   # Async: stream.out
  stream.out$sync()
  pipes.out.L2.L0[[i-2]]$run()          # Sync
  cuda.device.sync()

  # Iteration 12
  i <- 12
  pipes.out.L3.L2[[i %% 2 + 1]]$run()   # Async: stream.out
  stream.out$sync()
  pipes.out.L2.L0[[i-2]]$run()          # Sync
  cuda.device.sync()
}

# Run the processing to see how much time this takes, and compare it to the
# synchronous CUDA implementation, and the native R implementation
bench.CUDA.async <- microbenchmark( proc.CUDA.async(), times = 10 )
print( bench.R.sync )
print( bench.CUDA.sync )
print( bench.CUDA.async )

# Did we get the same result?
print( identical( tens.out.CUDA$ptr, tens.out.CUDA.async$ptr ) )

# The async process runtime is governed by the host memory speed in this case,
# hence (only) the 10x speedup compared to the native R implementation (w/ MRO).
# However, with more complex processing, or more processing steps, the memory
# transfer times can be completely hidden behind the processing. This would lead
# to the same speedup that we have seen when comparing just the GEMM
# implementations.

# Check out the async cuda process in NVVP!
# ./NVVP/007_NVVP_streaming.R

clean()
