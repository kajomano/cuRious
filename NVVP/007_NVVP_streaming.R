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

mat.in    <- matrix( round( rnorm( 10*n*n ) ), nrow = n, ncol = 10*n )
mat.proc  <- t( diag( 1, n, n ) )

# Tensors
tens.in        <- tensor$new( mat.in )
tens.out       <- tensor$new( mat.in, copy = FALSE )

tens.proc      <- tensor$new( mat.proc, 3L )

tens.in.stage  <- tensor$new( mat.proc, 2L, copy = FALSE )
tens.in.1      <- tensor$new( mat.proc, 3L, copy = FALSE )
tens.in.2      <- tensor$new( mat.proc, 3L, copy = FALSE )

tens.out.stage <- tensor$new( mat.proc, 2L, copy = FALSE )
tens.out.1     <- tensor$new( mat.proc, 3L, copy = FALSE )
tens.out.2     <- tensor$new( mat.proc, 3L, copy = FALSE )

# Streams
stream.in      <- stream$new( 3L )
stream.out     <- stream$new( 3L )
stream.proc    <- stream$new( 3L )

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
      gemms[[i %% 2 + 1]]$run()
      gemms[[i %% 2 + 1]]$run()
      gemms[[i %% 2 + 1]]$run()
    }

    stream.in$sync()
    stream.out$sync()
    stream.proc$sync()
  }
}

proc()
proc()
proc()

print("Finished")
