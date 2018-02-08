# Script to use with nvprof
# I had a lot of trouble with nvvp nad especially nvprof giving all kinds of
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

n <- 1000

mat.in             <- matrix( round(rnorm( 10*n*n )), nrow = n, ncol = 10*n )
mat.out.CUDA.async <- matrix( 0, nrow = n, ncol = 10*n )

mat.proc <- t(diag( 1, n, n ))

mat.dummy <- matrix( 0, n, n )

tens.in.1 <- tensor$new( mat.dummy, 3 )
tens.in.2 <- tensor$new( mat.dummy, 3 )

tens.proc <- tensor$new( mat.proc, 3 )

tens.out.1 <- tensor$new( mat.dummy, 3 )
tens.out.2 <- tensor$new( mat.dummy, 3 )

handle <- cublas.handle$new()$activate()

tens.in.stage  <- tensor$new( mat.dummy, 2 )
tens.out.stage <- tensor$new( mat.dummy, 2 )

stream.in   <- cuda.stream$new()$activate()
stream.proc <- cuda.stream$new()$activate()
stream.out  <- cuda.stream$new()$activate()

proc.CUDA.async <- function(){
  # Spool up
  # Iteration 1
  i <- 1

  # Create a column subset
  col.subset.in <- list( (1+(i-1)*n), (i*n) )

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
  col.subset.in <- list( (1+(i-1)*n), (i*n) )

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

    col.subset.in  <- list( (1+(i-1)*n), (i*n) )
    col.subset.out <- list( (1+(i-3)*n), ((i-2)*n) )

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
  col.subset.out <- list( (1+(i-3)*n), ((i-2)*n) )
  cublas.sgemm( handle, tens.in.2, tens.proc, tens.out.2, 1, 0, stream = stream.proc )
  transfer( tens.out.1, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, mat.out.CUDA.async, cols.dst = col.subset.out )

  cuda.stream.sync.all()

  # Iteration 12
  i <- 12
  col.subset.out <- list( (1+(i-3)*n), ((i-2)*n) )
  transfer( tens.out.2, tens.out.stage, stream = stream.out )
  cuda.stream.sync( stream.out )
  transfer( tens.out.stage, mat.out.CUDA.async, cols.dst = col.subset.out )

  cuda.stream.sync.all()
}

proc.CUDA.async()

print("Finished")
