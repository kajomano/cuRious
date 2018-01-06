# This script shows a simple matrix-matrix mutliplication (GEMM) using cuBLAS
library( cuRious )

# Create tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
m <- 6
n <- 4
k <- 5

mat.A <- matrix( 1:(m*k), ncol = k )
tens.A <- tensor$new( mat.A )
tens.A$dive()

mat.B <- matrix( 1:(k*n), ncol = n )
tens.B <- tensor$new( mat.B )
tens.B$dive()

mat.C <- matrix( 1:(m*n), ncol = n )
tens.C <- tensor$new( mat.C )
tens.C$dive()

# Create a cublas handle and add the two matrices, the result ending up in
# tens.C
handle <- cublas.handle$new()
handle$activate()
cublas.sgemm( handle, tens.A, tens.B, tens.C )

# Check if we got a correct result: it should be equal, as we used whole numbers
print( tens.C$pull() )
print( mat.A %*% mat.B + mat.C )

clean.global()
