library( cuRious )
library( microbenchmark )

# This script shows how to use streams and asynchronous pip transfers
# parallel with asychronous kernel launches, also known as data streaming. This
# technique is utilized to great effect to hide the latency of data transfers
# between the host and the device.

# With asynchronous memory transfer calls we can parallelize processing and
# memory transfers by double buffering incoming and outgoing pipes:
# Time --->
#
# L0       [tens.in]                       [tens.out]
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

# Lets define this operation, and run it deployed to different levels. This sort
# of dynamic deployment is a main design goal of cuRious:

# Base matrix size
n <- 1000

# Let's create a larger matrix that will serve as our input data. This matrix
# will be processed in 10 equal sized chunks
mat.in    <- matrix( round( rnorm( 10*n*n ) ), nrow = n, ncol = 10*n )

# This matrix will serve as our processing matrix for the gemm operation
mat.proc  <- t( diag( 1, n, n ) )

# Tensors
tens.in        <- tensor$new( mat.in )
tens.out       <- tensor$new( mat.in, copy = FALSE )

tens.proc      <- tensor$new( mat.proc )

tens.in.stage  <- tensor$new( mat.proc, copy = FALSE )
tens.in.1      <- tensor$new( mat.proc, copy = FALSE )
tens.in.2      <- tensor$new( mat.proc, copy = FALSE )

tens.out.stage <- tensor$new( mat.proc, copy = FALSE )
tens.out.1     <- tensor$new( mat.proc, copy = FALSE )
tens.out.2     <- tensor$new( mat.proc, copy = FALSE )

# Streams
stream.in      <- stream$new()
stream.out     <- stream$new()
stream.proc    <- stream$new()

# Pipe contexts
pipe.cont.in   <- pipe.context$new( stream.in )
pipe.cont.out  <- pipe.context$new( stream.out )

# Pipes
pipes.in.L0.L2 <- lapply( 1:10, function( i ){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.in, tens.in.stage, src.span = span, context = pipe.cont.in )
})

pipes.in.L2.L3 <- list(
  pipe$new( tens.in.stage, tens.in.1, context = pipe.cont.in ),
  pipe$new( tens.in.stage, tens.in.2, context = pipe.cont.in )
)

pipes.out.L3.L2 <- list(
  pipe$new( tens.out.1, tens.out.stage, context = pipe.cont.out ),
  pipe$new( tens.out.2, tens.out.stage, context = pipe.cont.out )
)

pipes.out.L2.L0 <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( tens.out.stage, tens.out, dst.span = span, context = pipe.cont.out )
})

# cuBLAS contexts
cublas.cont <- cublas.context$new( stream.proc )

# GEMM fusions
gemms <- list(
  cublas.sgemm$new( tens.in.2,
                    tens.proc,
                    tens.out.2,
                    alpha = 1,
                    beta = 0,
                    context = cublas.cont ),

  cublas.sgemm$new( tens.in.1,
                    tens.proc,
                    tens.out.1,
                    alpha = 1,
                    beta = 0,
                    context = cublas.cont )
)

proc <- function(){
  for( i in 1:11 ){
    if( i %in% 3:12 ){
      pipes.out.L3.L2[[i %% 2 + 1]]$run()
      pipes.out.L2.L0[[i-2]]$run()
    }

    if( i %in% 1:10 ){
      pipes.in.L0.L2[[i]]$run()
      pipes.in.L2.L3[[i %% 2 + 1]]$run()
    }

    if( i %in% 2:11 ){
      gemms[[i %% 2 + 1]]$run()
    }

    stream.in$sync()
    stream.out$sync()
    stream.proc$sync()
  }
}

# Let's run the processing on the native R implementation (L0):
bench.R    <- microbenchmark( proc(), times = 10 )
tens.out.R <- tensor$new( tens.out )

# Same process on the device, but not parallelized with data transfers:
tens.proc$level      <- 3L

tens.in.stage$level  <- 2L
tens.in.1$level      <- 3L
tens.in.2$level      <- 3L

tens.out.stage$level <- 2L
tens.out.1$level     <- 3L
tens.out.2$level     <- 3L

pipe.cont.in$level   <- 3L
pipe.cont.out$level  <- 3L

cublas.cont$level    <- 3L

bench.cuda.sync    <- microbenchmark( proc(), times = 100 )
tens.out.cuda.sync <- tensor$new( tens.out )

# And finally, parallel data transfers:
stream.in$level   <- 3L
stream.out$level  <- 3L
stream.proc$level <- 3L

bench.cuda.async    <- microbenchmark( proc(), times = 100 )
tens.out.cuda.async <- tensor$new( tens.out )

# Let's see the benchmarking results:
print( bench.R )
print( bench.cuda.sync )
print( bench.cuda.async )

# Are the results the same?
print( identical( tens.out.R$pull(), tens.out.cuda.sync$pull() ) )
print( identical( tens.out.R$pull(), tens.out.cuda.async$pull() ) )

# Check out the async streaming in NVVP!
# ./NVVP/007_NVVP_streaming.R

clean()
