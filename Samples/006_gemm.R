library( cuRious )
library( microbenchmark )

# GEMM stands for GEneral Matrix-Multiply, and is an often used operation in
# neural networks. This script shows a simple matrix-matrix mutliplication
# using cuBLAS, a cuda library that implements Basic Linear-Algebraic
# Subroutines (BLAS).

# Create matrix tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
m <- 6
n <- 4
k <- 5

mat.A <- matrix( as.double( 1:(m*k) ), ncol = k )
mat.B <- matrix( as.double( 1:(k*n) ), ncol = n )
mat.C <- matrix( as.double( 1:(m*n) ), ncol = n )

A <- tensor$new( mat.A, 3L )
B <- tensor$new( mat.B, 3L )
C <- tensor$new( mat.C, 3L )

# Cublas operations require a cublas handle to be supplied. These handles are
# context objects, and are implemented the same way as cuda streams in cuRious.
handle <- cublas.handle$new()

# Let's create a GEMM operation. Following the same logic as pipes, cublas lib-
# rary calls are wrapped in persistent cublas objects to minimize call overhead
# on frequently reused operations:
gemm <- cublas.sgemm$new( A, B, C, handle = handle )

# Let's mutliply the two matrices, the result ending up in C. We can check if we
# got a correct result: they should be equal, as we used whole numbers.
gemm$run()

print( C$pull() )
print( mat.A %*% mat.B + mat.C )

# This gemm operation ran on the device, however, all cublas calls implement
# fallbacks for L0 tensors implemented in native R. This makes it easier to
# benchmark the speedup gained by using the gpu, or to debug an application.

# Let's benchmark the operation, and compare it to the R implementation:
A.0 <- tensor$new( matrix( rnorm( 10^6 ), 10^3, 10^3 ) )
B.0 <- tensor$new( A.0 )
C.0 <- tensor$new( A.0 )

A.3 <- tensor$new( A.0, 3L )
B.3 <- tensor$new( A.3 )
C.3 <- tensor$new( A.3 )

gemm.0 <- cublas.sgemm$new( A.0, B.0, C.0, handle = handle )
gemm.3 <- cublas.sgemm$new( A.3, B.3, C.3, handle = handle )

# TODO ====
# C.0 gains read-only

print( microbenchmark( gemm.0$run(), times = 10 ) )

clean()
