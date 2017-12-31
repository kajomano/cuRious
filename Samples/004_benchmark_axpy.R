# This script benchmarks the simple addition on the GPU
library( cuRious )
library( microbenchmark )

# Create vectors and store them in the device memory
n <- 10^6
vect.x <- rep( 1, times = n )
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- rep( 2, times = n )
tens.y <- tensor$new( vect.y )
tens.y$dive()

# Create cuBLAS handle
handle <- cublas.handle$new()
handle$create()

# Define functions for a better microbenchmark print
R.daxpy    <- function(){ vect.x * 0.5 + vect.y }
cuda.saxpy <- function(){ cublas.saxpy( tens.x, tens.y, 0.5, handle ) }

# Check the speeds
microbenchmark( R.daxpy(),    times = 1000 )
microbenchmark( cuda.saxpy(), times = 1000 )

clean.global()
