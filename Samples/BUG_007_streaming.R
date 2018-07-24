library( cuRious )
library( microbenchmark )

# This script shows how to use streams and asynchronous pip transfers
# parallel with asychronous kernel launches, also known as data streaming. This
# technique is utilized to great effect to hide the latency of data transfers
# between the host and the device.

# Base matrix size
n <- 1000

# Let's create a larger matrix that will serve as our input data. This matrix
# will be processed in 10 equal sized chunks
mat.in    <- matrix( round( rep( 10:1, times = n*n ) ), nrow = n, ncol = 10*n )
# mat.in    <- matrix( round( rnorm( 10*n*n ) ), nrow = n, ncol = 10*n )

# This matrix will serve as our processing matrix for the gemm operation
# mat.proc  <- t( diag( 1, n, n ) )
mat.proc  <- diag( 1, n, n )

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

#  -------------------------------------------

a.tens.in        <- tensor$new( mat.in )
a.tens.out       <- tensor$new( mat.in, copy = FALSE )

a.tens.proc      <- tensor$new( mat.proc )

a.tens.in.stage  <- tensor$new( mat.proc, copy = FALSE )
a.tens.in.1      <- tensor$new( mat.proc, copy = FALSE )
a.tens.in.2      <- tensor$new( mat.proc, copy = FALSE )

a.tens.out.stage <- tensor$new( mat.proc, copy = FALSE )
a.tens.out.1     <- tensor$new( mat.proc, copy = FALSE )
a.tens.out.2     <- tensor$new( mat.proc, copy = FALSE )

# Streams
stream.in      <- stream$new()
stream.out     <- stream$new()
stream.proc    <- stream$new()

# Pipe contexts
pipe.cont.in   <- pipe.context$new()
pipe.cont.out  <- pipe.context$new()

a.pipe.cont.in   <- pipe.context$new( stream.in )
a.pipe.cont.out  <- pipe.context$new( stream.out )

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
  pipe$new( tens.out.stage, tens.out, dst.span = span, context = pipe.cont.in )
})

#  ---------------------

a.pipes.in.L0.L2 <- lapply( 1:10, function( i ){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( a.tens.in, a.tens.in.stage, src.span = span, context = a.pipe.cont.in )
})

a.pipes.in.L2.L3 <- list(
  pipe$new( a.tens.in.stage, a.tens.in.1, context = a.pipe.cont.in ),
  pipe$new( a.tens.in.stage, a.tens.in.2, context = a.pipe.cont.in )
)

a.pipes.out.L3.L2 <- list(
  pipe$new( a.tens.out.1, a.tens.out.stage, context = a.pipe.cont.out ),
  pipe$new( a.tens.out.2, a.tens.out.stage, context = a.pipe.cont.out )
)

a.pipes.out.L2.L0 <- lapply( 1:10, function(i){
  span <- c( (1+(i-1)*n), (i*n) )
  pipe$new( a.tens.out.stage, a.tens.out, dst.span = span, context = a.pipe.cont.in )
})

# cuBLAS contexts
cublas.cont <- cublas.context$new()
a.cublas.cont <- cublas.context$new( stream.proc )

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

a.gemms <- list(
  cublas.sgemm$new( a.tens.in.2,
                    a.tens.proc,
                    a.tens.out.2,
                    alpha = 1,
                    beta = 0,
                    context = a.cublas.cont ),

  cublas.sgemm$new( a.tens.in.1,
                    a.tens.proc,
                    a.tens.out.1,
                    alpha = 1,
                    beta = 0,
                    context = a.cublas.cont )
)

proc <- function(){
  for( i in 1:3 ){
    if( i %in% 1:10 ){
      pipes.in.L0.L2[[i]]$run()
      pipes.in.L2.L3[[i %% 2 + 1]]$run()
    }

    if( i %in% 2:11 ){
      gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
    }

    if( i %in% 3:12 ){
      pipes.out.L3.L2[[i %% 2 + 1]]$run()
      pipes.out.L2.L0[[i-2]]$run()
    }
  }
}

a.proc <- function(){
  for( i in 1:3 ){
    if( i %in% 1:10 ){
      a.pipes.in.L0.L2[[i]]$run()
      a.pipes.in.L2.L3[[i %% 2 + 1]]$run()
    }

    if( i %in% 2:11 ){
      a.gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
      # gemms[[i %% 2 + 1]]$run()
    }

    if( i %in% 3:12 ){
      a.pipes.out.L3.L2[[i %% 2 + 1]]$run()
      a.pipes.out.L2.L0[[i-2]]$run()
    }

    stream.in$sync()
    stream.out$sync()
    stream.proc$sync()
  }
}

tens.proc$level      <- 3L

tens.in.stage$level  <- 2L
tens.in.1$level      <- 3L
tens.in.2$level      <- 3L

tens.out.stage$level <- 2L
tens.out.1$level     <- 3L
tens.out.2$level     <- 3L

pipe.cont.in$deploy( 3L )
pipe.cont.out$deploy( 3L )

cublas.cont$deploy( 3L )

stream.in$deploy( 3L )
stream.out$deploy( 3L )
stream.proc$deploy( 3L )

a.tens.proc$level      <- 3L

a.tens.in.stage$level  <- 2L
a.tens.in.1$level      <- 3L
a.tens.in.2$level      <- 3L

a.tens.out.stage$level <- 2L
a.tens.out.1$level     <- 3L
a.tens.out.2$level     <- 3L

a.pipe.cont.in$deploy( 3L )
a.pipe.cont.out$deploy( 3L )

a.cublas.cont$deploy( 3L )

proc()
a.proc()

# microbenchmark( proc() )
# microbenchmark( a.proc() )

print( identical( tens.out$pull(), a.tens.out$pull() ) )
print( which( tens.out$pull() != a.tens.out$pull(), arr.ind = T )[ 1:10, ] )

# INFO EDDIG:
# 125 tobbszoroseinel kezd el nem stimmelni mindig
# Az utso L2-L3 transfer hal meg
# MEGVAN: El vannak baszva a rangekben a +1-ek a dispatchnél

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

print( "Tests" )

print( identical( tens.in$pull(), a.tens.in$pull() ) )
print( identical( tens.in.stage$pull(), a.tens.in.stage$pull() ) )
print( identical( tens.in.1$pull(), a.tens.in.1$pull() ) )
print( identical( tens.in.2$pull(), a.tens.in.2$pull() ) )
print( identical( tens.out.1$pull(), a.tens.out.1$pull() ) )
print( identical( tens.out.2$pull(), a.tens.out.2$pull() ) )
print( identical( tens.out.stage$pull(), a.tens.out.stage$pull() ) )
print( identical( tens.out$pull(), a.tens.out$pull() ) )

