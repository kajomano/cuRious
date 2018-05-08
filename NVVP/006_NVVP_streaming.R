# Script to use with nvprof
# I had a lot of trouble with nvvp and, especially nvprof giving all kinds of
# strange segfaults and errors on linux. Turns out, you need to have sudo
# privileges when running nvprof, but then nothing is there on the PATH. The
# following worked:
# sudo /usr/local/cuda/bin/nvprof /usr/lib/R/bin/Rscript ./005_NVVP_streaming.R

print( paste(
  "sudo",
  "/usr/local/cuda/bin/nvvp",
  paste0( R.home(), "/bin/Rscript" ),
  paste0( getwd(), "/NVVP/005_NVVP_streaming.R" )
) )

library( cuRious )
library( microbenchmark )

# The main goal with streaming is to reduce memory consumption when handling
# large data sets, but also to hide the transfer latency behind the computation.
# Large datasets are split into smaller chunks by column subsetting, explained
# in an earlier sample script.

# Base matrix size
n <- 1000

# Let's create a larger matrix that will serve as our input data, and another
# one that will store the output:
mat.in    <- matrix( round(rnorm( 10*n*n )), nrow = n, ncol = 10*n )
mat.out.R <- matrix( 0, nrow = n, ncol = 10*n )

# This matrix will serve as our processing matrix
mat.proc <- t(diag( 1, n, n ))

tens.in       <- tensor$new( mat.in, 0L )
tens.out.CUDA <- tensor$new( NULL, 0L, obj.dims( mat.out.R ) )

tens.in.1  <- tensor$new( NULL, 3L, c( n, n ) )
tens.proc  <- tensor$new( mat.proc, 3L )
tens.out.1 <- tensor$new( NULL, 3L, c( n, n ) )

# cuBLAS handle
handle <- cublas.handle$new()

tens.out.CUDA.async <- tensor$new( NULL, 0L, obj.dims( mat.out.R ) )

tens.in.stage  <- tensor$new( NULL, 2L, c( n, n ) )
tens.out.stage <- tensor$new( NULL, 2L, c( n, n ) )

tens.in.2  <- tensor$new( NULL, 3L, c( n, n ) )
tens.out.2 <- tensor$new( NULL, 3L, c( n, n ) )

stream.in   <- cuda.stream$new()
stream.proc <- cuda.stream$new()
stream.out  <- cuda.stream$new()

proc.CUDA.async <- function(){
  # Spool up
  # Iteration 1
  i <- 1

  # Create a column subset
  col.subset.in <- c( (1+(i-1)*n), (i*n) )

  # Subset the input matrix and move to the stage (level 2 tensor), and then to
  # the device memory. We could do this in one step, but later this will be two,
  # so let's keep this also complicated!
  transfer( tens.in, tens.in.stage, src.cols = col.subset.in )

  # Async call, even though we don't need it just yet
  transfer( tens.in.stage, tens.in.1, stream = stream.in )

  # Wait for everything to finish
  cuda.stream.sync.all()

  # Iteration 2
  i <- 2
  col.subset.in <- c( (1+(i-1)*n), (i*n) )

  # Start processing tens.in.1 right away
  cublas.sgemm( tens.in.1, tens.proc, tens.out.1, alpha = 1, beta = 0, handle = handle, stream = stream.proc )

  # Move new data to the device
  transfer( tens.in, tens.in.stage, src.cols = col.subset.in )
  transfer( tens.in.stage, tens.in.2, stream = stream.in )
  cuda.stream.sync.all()

  # Iterations 3-10
  for( i in 3:10 ){
    # Tick or tock (double buffering)
    if( i %% 2 ){
      tens.in.cur   <- tens.in.1
      tens.out      <- tens.out.1
      tens.in.proc  <- tens.in.2
      tens.out.proc <- tens.out.2
    }else{
      tens.in.cur   <- tens.in.2
      tens.out      <- tens.out.2
      tens.in.proc  <- tens.in.1
      tens.out.proc <- tens.out.1
    }

    col.subset.in  <- c( (1+(i-1)*n), (i*n) )
    col.subset.out <- c( (1+(i-3)*n), ((i-2)*n) )

    # Start processing
    cublas.sgemm( tens.in.proc, tens.proc, tens.out.proc, alpha = 1, beta = 0, handle = handle, stream = stream.proc )

    # Start moving data out (async part), and in (sync part)
    transfer( tens.out, tens.out.stage, stream = stream.out )
    transfer( tens.in, tens.in.stage, src.cols = col.subset.in )

    # Start moving data in (async part), and out (sync part)
    transfer( tens.in.stage, tens.in.cur, stream = stream.in )
    cuda.stream.sync( stream.out )
    transfer( tens.out.stage, tens.out.CUDA.async, dst.cols = col.subset.out )

    cuda.stream.sync.all()
  }

  # Spool down
  # Iteration 11
  i <- 11
  col.subset.out <- c( (1+(i-3)*n), ((i-2)*n) )
  cublas.sgemm( tens.in.2, tens.proc, tens.out.2, alpha = 1, beta = 0, handle = handle, stream = stream.proc )
  transfer( tens.out.1, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, tens.out.CUDA.async, dst.cols = col.subset.out )

  cuda.stream.sync.all()

  # Iteration 12
  i <- 12
  col.subset.out <- c( (1+(i-3)*n), ((i-2)*n) )
  transfer( tens.out.2, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, tens.out.CUDA.async, dst.cols = col.subset.out )

  cuda.stream.sync.all()
}

proc.CUDA.async()


print("Finished")
