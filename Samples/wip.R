library( cuRious )
library( microbenchmark )

# C = A( m, k ) %*% B( k, n ) + C( m, n )
m     <- 6
n     <- 5

vect.x <- rep( 1, times = m )
vect.y <- rep( 1, times = n )
mat.A  <- matrix( as.numeric( 1:(n*m) + 10 ), ncol = n, nrow = m )

tens.x <- tensor$new( vect.x )
tens.y <- tensor$new( vect.y )
tens.A <- tensor$new( mat.A )

tens.x.under <- duplicate.obj( tens.x )$dive()
tens.y.under <- duplicate.obj( tens.y )$dive()
tens.A.under <- duplicate.obj( tens.A )$dive()

handle <- cublas.handle$new()$activate()

cublas.sger( tens.x, tens.y, tens.A, alpha = 2 )
cublas.sger( tens.x.under, tens.y.under, tens.A.under, alpha = 2, handle = handle )

print( tens.A$pull() )
print( tens.A.under$pull() )

identical( tens.A$pull(), tens.A.under$pull() )
