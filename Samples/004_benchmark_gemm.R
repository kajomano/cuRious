library( cuRious )
library( microbenchmark )

# Create tensors and store them in GPU memory
# A( m, k ) * B( k, n ) + C( m, n )
m <- 2000
n <- 1000
k <- 1500

mat.A <- matrix( rnorm(m*k), ncol = k )
tens.A <- tensor$new( mat.A )
tens.A$dive()

mat.B <- matrix( rnorm(k*n), ncol = n )
tens.B <- tensor$new( mat.B )
tens.B$dive()

mat.C <- matrix( rnorm(m*n), ncol = n )
tens.C <- tensor$new( mat.C )
tens.C$dive()

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle
handle <- cublas.handle$new()
handle$activate()

# Define functions for a better microbenchmark print
# cuBLAS calls are asynchronous, for a proper timing we need to call sync.streams()
R.dgemm <- function(){
  mat.C <<- ( mat.A %*% mat.B ) * alpha + mat.C * beta
}
cuda.sgemm <- function(){
  cublas.sgemm( handle, tens.A, tens.B, tens.C )
}

# Check the speeds
print( microbenchmark( R.dgemm(),    times = 10 ) )
print( microbenchmark( cuda.sgemm(), times = 10 ) )

clean.global()
