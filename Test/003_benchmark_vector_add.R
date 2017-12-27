# This script benchmarks the simple addition on the GPU
library( cuRious )
library( microbenchmark )

# Create vectors and store them in the device memory
n <- 10^6
vect.x <- rep( 1, times = n )
vect.x.obj   <- vect$new( vect.x )
vect.x.obj$dive()

vect.y <- rep( 2, times = n )
vect.y.obj   <- vect$new( vect.y )
vect.y.obj$dive()

vect.res.obj <- vect$new( vect.y )
vect.res.obj$dive()

# Define functions for a better microbenchmark print
R.add <- function(){ vect.x + vect.y }
cuda.add <- function(){ vect.add( vect.x.obj, vect.y.obj, vect.res.obj ) }

# Check the speeds
microbenchmark( R.add(), times = 100 )
microbenchmark( cuda.add(), times = 100 )
