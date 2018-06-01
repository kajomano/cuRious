# Script to use with nvprof
# I had a lot of trouble with nvvp and especially nvprof giving all kinds of
# strange segfaults and errors on linux. Turns out, you need to have sudo
# privileges when running nvprof, but then nothing is there on the PATH. The
# following worked:
# sudo /usr/local/cuda/bin/nvprof /usr/lib/R/bin/Rscript ./007_NVVP_streaming.R

print( paste(
  "sudo",
  "/usr/local/cuda/bin/nvvp",
  paste0( R.home(), "/bin/Rscript" ),
  paste0( getwd(), "/NVVP/007_NVVP_streaming.R" )
) )

library( cuRious )

# Base matrix size
n <- 1000

mat.in   <- matrix( round(rnorm( 10*n*n )), nrow = n, ncol = 10*n )
mat.proc <- t(diag( 1, n, n ))

tens.in       <- tensor$new( mat.in, 0L )

tens.in.1  <- tensor$new( NULL, 3L, c( n, n ) )
tens.proc  <- tensor$new( mat.proc, 3L )
tens.out.1 <- tensor$new( NULL, 3L, c( n, n ) )

handle <- cublas.handle$new()

tens.out.CUDA.async <- tensor$new( tens.in, copy = FALSE )

# Asynchronous transfers require L2 staging buffers:
tens.in.stage  <- tensor$new( NULL, 2L, c( n, n ) )
tens.out.stage <- tensor$new( NULL, 2L, c( n, n ) )

tens.in.2  <- tensor$new( NULL, 3L, c( n, n ) )
tens.out.2 <- tensor$new( NULL, 3L, c( n, n ) )

stream.in   <- cuda.stream$new()
stream.proc <- cuda.stream$new()
stream.out  <- cuda.stream$new()

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

proc.CUDA.async()

clean()

print("Finished")
