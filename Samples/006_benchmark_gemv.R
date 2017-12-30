# Gemv is a more involved computation than axpy, so we should see a bigger
# speedup here
library( cuRious )
library( microbenchmark )

# Create tensors and store them in memory
# A( nrow, ncol ) * x( ncol ) + y( nrow )
n.row <- 10000
n.col <- 1000

mat.A <- matrix( rnorm( n.row*n.col ), ncol = n.col )
tens.A <- tensor$new( mat.A )
tens.A$dive()

vect.x <- rnorm( n.col )
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- rnorm( n.row )
tens.y <- tensor$new( vect.y )
tens.y$dive()

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()

# Define functions for a better microbenchmark print
R.dgemv    <- function(){ ( mat.A %*% vect.x ) * alpha + vect.y * beta }
cuda.sgemv <- function(){ cublas.sgemv( tens.A, tens.x, tens.y, alpha, beta, 'N', handle ) }

# Check the speeds
microbenchmark( R.dgemv(),    times = 1000 )
microbenchmark( cuda.sgemv(), times = 1000 )
