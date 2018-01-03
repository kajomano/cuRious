library( cuRious )
library( microbenchmark )

n <- 1000
mat <- matrix( rnorm(n*n), ncol = n )

tens.A <- tensor$new( mat )
tens.A$dive()

tens.B <- tensor$new( mat )
tens.B$dive()

tens.C <- tensor$new( mat )
tens.C$dive()

# Create a cublas handle and two streams
handle <- cublas.handle$new()
handle$create()
stream1 <- cuda.stream$new()
stream1$create()
stream2 <- cuda.stream$new()
stream2$create()

# Make tens.A staged, a it will be used frequently for CPU<-->GPU data transfers
# Asynchronous data transfers are only possible with staged tensors!
tens.A$create.stage()

# Define functions
single.stream <- function(){
  tens.A$push( mat )
  cublas.sgemm( handle, tens.B, tens.B, tens.C, 1, 0 )
  cuda.streams.sync()
}

# multiple.streams <- function(){
#   tens.A$push( mat )
#   cublas.sgemm( handle, tens.B, tens.B, tens.C, 1, 0 )
#   cuda.streams.sync()
# }

# Check the speeds
microbenchmark( single.stream(),    times = 10 )
# microbenchmark( multiple.streams(), times = 1 )

clean.global()
