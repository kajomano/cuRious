# This script shows a simple addition operation on the GPU
library( cuRious )

# Create vectors and store them in the device memory
n <- 10
vect.x <- rnorm( n )
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- rnorm( n )
tens.y <- tensor$new( vect.y )
tens.y$dive()
tens.y$pull()

# Create a vector in the device memory to store the results
vect.res <- rep( 0, times = n )
tens.res <- tensor$new( vect.res )
tens.res$dive()

# Add the two vectors
ewop( tens.x, tens.y, tens.res )

# Check if we got a correct result: it should be mostly equal :D
tens.res$pull()
print( vect.x + vect.y )

