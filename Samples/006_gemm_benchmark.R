# This script compares the speeds of a native R GEMM routine with the cuRious
# implementation that uses cuBLAS.
library( cuRious )
library( microbenchmark )

# Create tensors and store them in GPU memory
# C = A( m, k ) * B( k, n ) * alpha + C( m, n ) * beta
m <- 2000
n <- 1000
k <- 1500

mat.A <- matrix( rnorm(m*k), ncol = k )
mat.B <- matrix( rnorm(k*n), ncol = n )
mat.C <- matrix( rnorm(m*n), ncol = n )

tens.A <- tensor$new( mat.A, 3 )
tens.B <- tensor$new( mat.B, 3 )
tens.C <- tensor$new( mat.C, 3 )

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle
handle <- cublas.handle$new()

# Define functions for a better microbenchmark print
R.dgemm <- function(){
  mat.C <<- ( mat.A %*% mat.B ) * alpha + mat.C * beta
}

cuda.sgemm <- cublas.sgemm$new( tens.A, tens.B, tens.C, handle = handle )

# Check the speeds
# The R.dgemm() benchmark can take quite a while if you don't have MRO installed
# that is why we only do 10 iterations of the measurement here
print( microbenchmark( R.dgemm(),        times = 10 ) )
print( microbenchmark( cuda.sgemm$run(), times = 10 ) )

clean()
