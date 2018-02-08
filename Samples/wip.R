library( cuRious )
library( microbenchmark )

# C = A( m, k ) %*% B( k, n ) + C( m, n )
m     <- 6
n     <- 5
k     <- 4
sub.k <- 2

mat.A <- matrix( 1, ncol = k, nrow = m )
mat.B <- matrix( 1, ncol = n, nrow = sub.k )
mat.C <- matrix( 0, ncol = n, nrow = m )

tens.A <- tensor$new( mat.A )
tens.B <- tensor$new( mat.B )
tens.C <- tensor$new( mat.C )

tens.A.under <- duplicate.obj( tens.A )$dive()
tens.B.under <- duplicate.obj( tens.B )$dive()
tens.C.under <- duplicate.obj( tens.C )$dive()

handle <- cublas.handle$new()$activate()

sub   <- list(1+1, sub.k+1)

cublas.sgemm( tens.A, tens.B, tens.C, sub, sub, sub )
cublas.sgemm( tens.A.under, tens.B.under, tens.C.under, sub, sub, sub, handle = handle )

print( tens.C$pull() )
print( tens.C.under$pull() )

identical( tens.C$pull(), tens.C.under$pull() )
