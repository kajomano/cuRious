# This script benchmarks a simple element-wise addition. We compare the native R
# implementation to the cuBLAS and our own custom kernel. Custom kernels that do
# element-wise operations will be very important in neural networks, as all of
# the activation functions will be implemented with these.

library( cuRious )
library( microbenchmark )

n <- 10^6
vect.x <- rep( 1, times = n )
vect.y <- rep( 0, times = n )

# Create tensors
tens.x.cub <- tensor$new( vect.x, 3 )
tens.y.cub <- tensor$new( vect.y, 3 )

tens.x.alg <- tensor$new( vect.x, 3 )
tens.y.alg <- tensor$new( vect.y, 3 )

# Create cuBLAS handle
handle <- cublas.handle$new()

# Define functions for a better microbenchmark print
R.daxpy      <- function(){ vect.x * 0.5 + vect.y }
cuda.saxpy   <- function(){ cublas.saxpy( handle, tens.x.cub, tens.y.cub, 0.5 ) }
custom.saxpy <- function(){ alg.saxpy( tens.x.alg, tens.y.alg, 0.5 ) }

# Check the speeds
print( microbenchmark( R.daxpy(),      times = 1000 ) )
print( microbenchmark( cuda.saxpy(),   times = 1000 ) )
print( microbenchmark( custom.saxpy(), times = 1000 ) )

# As can be seen, the custom kernel is as fast, or even faster than the cuBLAS
# implementation.

# Are the results the same?
print( identical( tens.y.cub$pull(), tens.y.alg$pull() ) )

clean.global()
