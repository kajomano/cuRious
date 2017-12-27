# This script shows a simple addition operation on the GPU
library( cuRious )

# Create vectors and store them in the device memory
n <- 10
vect.x <- rnorm( n )
vect.x.obj <- vect$new( vect.x )
vect.x.obj$dive()

vect.y <- rnorm( n )
vect.y.obj <- vect$new( vect.y )
vect.y.obj$dive()

# Create a vector in the device memory to store the results
vect.res <- rep( 0, times = n )
vect.res.obj <- vect$new( vect.res )
vect.res.obj$dive()

# Add the two vectors
vect.add( vect.x.obj, vect.y.obj, vect.res.obj )

# Check if we got a correct result: it should be mostly equal :D
vect.res.obj$pull()
print( vect.x + vect.y )

