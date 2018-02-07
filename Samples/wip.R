library( cuRious )
library( microbenchmark )

# C = A( m, k ) %*% B( k, n ) + C( m, n )
m <- 6
n <- 4
k <- 5

mat.A <- matrix( 1, ncol = k, nrow = m )
mat.B <- matrix( 1, ncol = n, nrow = k )
mat.C <- matrix( 1, ncol = n, nrow = m )

tens.A <- tensor$new( mat.A )
tens.B <- tensor$new( mat.B )
tens.C <- tensor$new( mat.C )

tens.A.under <- duplicate.obj( tens.A )$dive()
tens.B.under <- duplicate.obj( tens.B )$dive()
tens.C.under <- duplicate.obj( tens.C )$dive()

handle <- cublas.handle$new()$activate()

cublas.sgemm( tens.A, tens.B, tens.C )
cublas.sgemm( tens.A.under, tens.B.under, tens.C.under, handle = handle )

identical( tens.C$pull(), tens.C.under$pull() )
