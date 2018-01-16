library( cuRious )
library( microbenchmark )

n <- 10^6
vect.x <- rep( 0, times = n )

tens.x <- tensor$new( vect.x )
tens.x$dive()

handle <- cublas.handle$new()
handle$activate()

stream <- cuda.stream$new()
stream$activate()

stream2 <- cuda.stream$new()
stream2$activate()

for( i in 1:30 ){
  cublas.saxpy( handle, tens.x, tens.x, 0.5, stream )
  alg.saxpy( tens.x, tens.x, 0.5, stream2 )
}

# clean.global()
