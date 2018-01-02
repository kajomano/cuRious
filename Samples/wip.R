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

# Make tens.A staged. This creates a page-locked buffer for the tensor
# that can accelerate all push, pull, dive and surface calls for the cost
# of taking up memory in non-swappable address space. This is useful for
# tensors that will act as IO points between the CPU and GPU with regular
# push or pull operations. Use sparingly!
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
