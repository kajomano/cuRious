# This script is for profiling the parallel cudaMemcpyAsync() calls with NVVP
library( cuRious )
library( microbenchmark )

n     <- 1000
mat   <- matrix( as.double(1:(n*n)), ncol = n )
tens1 <- tensor$new( mat )
tens2 <- tensor$new( mat )

tens1$create.stage()
tens1$dive()

tens2$create.stage()
tens2$dive()

stream1 <- cuda.stream$new()
stream1$activate()

stream2 <- cuda.stream$new()
stream2$activate()

par.copy <- function(){
  for( i in 1:10 ){
    tens1$pull.prefetch.async( stream1 )
    tens2$push.preproc( mat )
    tens2$push.fetch.async( stream2 )
    cuda.stream.sync( stream1 )
    tens1$pull.proc()
    cuda.stream.sync( stream2 )
  }
}

par.copy()

print("Finished")
