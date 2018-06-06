library( cuRious )

# Create matrix tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
cols <- 10
rows <- 6
subs <- c( 2, 7 )

mat.A <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.B <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.C <- matrix( as.double( 1:(cols*rows) ), ncol = cols )

tens.A.3 <- tensor$new( mat.A , 3 )
tens.B.3 <- tensor$new( mat.B , 3 )
tens.C.3 <- tensor$new( mat.C , 3 )

tens.A.0 <- tensor$new( mat.A , 0 )
tens.B.0 <- tensor$new( mat.B , 0 )
tens.C.0 <- tensor$new( mat.C , 0 )

handle <- cublas.handle$new()

L3.sgemm <- cublas.sgemm$new( tens.A.3, tens.B.3, tens.C.3, subs, subs, subs, FALSE, TRUE, handle = handle )
L0.sgemm <- cublas.sgemm$new( tens.A.0, tens.B.0, tens.C.0, subs, subs, subs, FALSE, TRUE )

L3.sgemm$run()
L0.sgemm$run()

print( tens.C.3$pull() )
print( tens.C.0$pull() )

clean()
