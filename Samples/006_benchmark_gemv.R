library( cuRious )
library( microbenchmark )

# Create tensors and store them in GPU memory
# A( m, n ) * x( n ) + y( m)
m <- 1500
n <- 1000

mat.A <- matrix( 1:(m*n), ncol = n )
tens.A <- tensor$new( mat.A )
tens.A$dive()

vect.x <- 1:n
tens.x <- tensor$new( vect.x )
tens.x$dive()

vect.y <- 1:m
tens.y <- tensor$new( vect.y )
tens.y$dive()

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()
handle$create()

# Define functions for a better microbenchmark print
R.dgemv    <- function(){ ( mat.A %*% vect.x ) * alpha + vect.y * beta }
cuda.sgemv <- function(){ cublas.sgemv( tens.A, tens.x, tens.y, alpha, beta, 'N', handle ) }

# Check the speeds
microbenchmark( R.dgemv(),    times = 1000 )
microbenchmark( cuda.sgemv(), times = 1000 )

clean.global()
