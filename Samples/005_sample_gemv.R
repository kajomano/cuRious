# This script shows a simple matrix-vector mutliplication
library( cuRious )

# Create tensors and store them in memory
# A( nrow, ncol ) * x( ncol ) + y( nrow )
n.row <- 3
n.col <- 2

mat.A <- matrix( 1:(n.row*n.col), ncol = n.col )
tens.A <- tensor$new( mat.A )
tens.A$dive()

vect.x <- 1:n.col
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- 1:n.row
tens.y <- tensor$new( vect.y )
tens.y$dive()

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()
cublas.sgemv( tens.A, tens.x, tens.y, 1, 1, 'N', handle )

# Check if we got a correct result: it should be equal, as we used whole numbers
tens.y$pull()
print( as.vector( mat.A %*% vect.x + vect.y ) )
