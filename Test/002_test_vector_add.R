# This script shows how to move vectors to/from the GPU memory
library( cuRious )

# Create vectors and store them in the device memory
n <- 10
vect.x <- rnorm( n ) * 10^9
vect.y <- rnorm( n ) * 10^9
vect.x.ptr <- dive( vect.x )
vect.y.ptr <- dive( vect.y )



'%+%'
