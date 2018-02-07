# This script shows a simple matrix-matrix mutliplication (GEMM) using cuBLAS,
# a cuda library that implements Basic Linear-Algebraic Subroutines (BLAS).
# GEMM stands for GEneral Matrix-Multiply, and is an often used function in
# neural networks
library( cuRious )

# Create matrix tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
m <- 6
n <- 4
k <- 5

mat.A <- matrix( as.double( 1:(m*k) ), ncol = k )
mat.B <- matrix( as.double( 1:(k*n) ), ncol = n )
mat.C <- matrix( as.double( 1:(m*n) ), ncol = n )

tens.A <- tensor$new( mat.A )$dive()
tens.B <- tensor$new( mat.B )$dive()
tens.C <- tensor$new( mat.C )$dive()

# Create a cuBLAS handle and activate it. An activated cuBLAS handle is needed
# for each cuBLAS call. As it is costly to create a handle, it is advised to
# reuse the handle throughout multiple calls, or even the whole session.
handle <- cublas.handle$new()$activate()

# Mutliply the two matrices, the result ending up in tens.C
cublas.sgemm( handle, tens.A, tens.B, tens.C )

# Check if we got a correct result: they should be equal, as we used whole numbers
print( tens.C$pull() )
print( mat.A %*% mat.B + mat.C )

clean.global()
