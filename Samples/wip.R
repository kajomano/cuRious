library( cuRious )
library( microbenchmark )

n     <- 10000
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

async.transfer <- function(){
  tens1$pull.prefetch.async( stream1 )
  tens2$push.preproc( mat )
  tens2$push.fetch.async( stream2 )
  cuda.stream.sync( stream1 )
  tens1$pull.proc()
  cuda.stream.sync( stream2 )
}

sync.transfer <- function(){
  tens1$pull()
  tens2$push( mat )
}

microbenchmark( sync.transfer(), times = 10 )
microbenchmark( async.transfer(), times = 10 )

clean.global()
