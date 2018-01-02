library( cuRious )
library( microbenchmark )

# Create tensors and store them in GPU memory
# A( m, k ) * B( k, n ) + C( m, n )
m <- 1000
n <- 1500
k <- 2000

mat.A <- matrix( rnorm(m*k), ncol = k )
tens.A <- tensor$new( mat.A )
tens.A$dive()

mat.B <- matrix( rnorm(k*n), ncol = n )
tens.B <- tensor$new( mat.B )
tens.B$dive()

mat.C <- matrix( rnorm(m*n), ncol = n )
tens.C <- tensor$new( mat.C )
tens.C$dive()

alpha <- -1.5
beta  <- 0.5

# Create a cublas handle and add the two vectors, the result ending up in tens.y
handle <- cublas.handle$new()
handle$create()

# Define functions for a better microbenchmark print
R.dgemm    <- function(){
  for( i in 1:10 ){
    mat.C <<- ( mat.A %*% mat.B ) * alpha + mat.C * beta
  }
}
cuda.sgemm <- function(){
  for( i in 1:10 ){
    cublas.sgemm( tens.A, tens.B, tens.C, alpha, beta, handle = handle )
  }
  tens.C$pull()
}

# Check the speeds
microbenchmark( R.dgemm(),    times = 1 )
microbenchmark( cuda.sgemm(), times = 1 )

clean.global()
