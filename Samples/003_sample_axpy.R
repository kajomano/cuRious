# This script shows a simple vector addition operation on the GPU
library( cuRious )

# Create vectors and store them in the device memory
n <- 10

vect.x <- rnorm( n )
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- rnorm( n )
tens.y <- tensor$new( vect.y )
tens.y$dive()

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()
handle$create()
cublas.saxpy( tens.x, tens.y, 1, handle )

# Check if we got a correct result: it should be mostly equal
tens.y$pull()
print( vect.x + vect.y )

clean.global()
