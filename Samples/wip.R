library( cuRious )
library( microbenchmark )

n     <- 1000
mat   <- matrix( 1:n*n, ncol = n )
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

microbenchmark( tens1$pull(), times = 100 )
microbenchmark( tens1$pull.proc(), times = 100 )

microbenchmark( tens2$push( mat ), times = 100 )
microbenchmark( tens2$push.preproc( mat ), times = 100 )

microbenchmark( sync.transfer(), times = 100 )
profvis::profvis( sync.transfer(), interval = 0.001 )
microbenchmark( async.transfer(), times = 100 )

clean.global()
