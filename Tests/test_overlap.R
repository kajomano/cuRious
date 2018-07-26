source( "./Tests/test_utils.R" )

n       <- 1000
times   <- 100

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
stream.in      <- stream$new()
stream.out     <- stream$new()
stream.proc    <- stream$new()

# Pipe contexts
pipe.cont.in   <- pipe.context$new( stream.in,  3L )
pipe.cont.out  <- pipe.context$new( stream.out, 3L )

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
cublas.cont <- cublas.context$new( stream.proc, 3L )

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
    }

    stream.in$sync()
    stream.out$sync()
    stream.proc$sync()
  }
}

bench.cuda.sync    <- microbenchmark( proc(), times = times )
tens.out.cuda.sync <- tensor$new( tens.out )

stream.in$level   <- 3L
stream.out$level  <- 3L
stream.proc$level <- 3L

bench.cuda.async    <- microbenchmark( proc(), times = times )
tens.out.cuda.async <- tensor$new( tens.out )

if( !identical( tens.out.cuda.sync$pull(), tens.out.cuda.async$pull() ) ){
  stop( "Non-identical results" )
}

print( paste0( "sync: ", min( bench.cuda.sync$time ) / 10^6, " ms" ) )
print( paste0( "async: ", min( bench.cuda.async$time ) / 10^6, " ms" ) )

clean()
