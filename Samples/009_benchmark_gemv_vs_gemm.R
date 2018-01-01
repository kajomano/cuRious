library( cuRious )
library( microbenchmark )

# Create tensors and store them in GPU memory
# A( m, n ) * x( n ) + y( m)
m <- 15000
n <- 1000

mat.A <- matrix( 1:(m*n), ncol = n )
tens.A <- tensor$new( mat.A )
tens.A$dive()

vect.x <- 1:n
tens.x <- tensor$new( vect.x )
tens.x$dive()

mat.X <- matrix( 1:n, ncol = 1 )
tens.X <- tensor$new( mat.X )
tens.X$dive()

vect.y <- 1:m
tens.y <- tensor$new( vect.y )
tens.y$dive()

mat.Y <- matrix( 1:m, ncol = 1 )
tens.Y <- tensor$new( mat.Y )
tens.Y$dive()

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()
handle$create()

# Define functions for a better microbenchmark print
cuda.sgemv <- function(){ cublas.sgemv( tens.A, tens.x, tens.y, alpha, beta, 'N', handle ) }
cuda.sgemm <- function(){ cublas.sgemm( tens.A, tens.X, tens.Y, alpha, beta, 'N', 'N', handle ) }

# Check the speeds
microbenchmark( cuda.sgemv(), times = 1000 )
microbenchmark( cuda.sgemm(), times = 1000 )

clean.global()
