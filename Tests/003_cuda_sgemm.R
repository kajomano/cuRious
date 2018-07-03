library( cuRious )
library( microbenchmark )

# Create matrix tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
# mult <- 1

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.B <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.C <- matrix( as.double( 1:(cols*rows) ), ncol = cols )

tens.A.3 <- tensor$new( mat.A , 3 )
tens.B.3 <- tensor$new( mat.B , 3 )
tens.C.3 <- tensor$new( mat.C , 3 )

tens.A.0 <- tensor$new( mat.A , 0 )
tens.B.0 <- tensor$new( mat.B , 0 )
tens.C.0 <- tensor$new( mat.C , 0 )

unit.A.3 <- tensor$new( 1.0, 3 )
unit.B.3 <- tensor$new( 1.0, 3 )
unit.C.3 <- tensor$new( 1.0, 3 )

L3.sgemm <- cublas.sgemm$new( tens.A.3, tens.B.3, tens.C.3, subs, subs, subs, FALSE, TRUE, context = context )
L0.sgemm <- cublas.sgemm$new( tens.A.0, tens.B.0, tens.C.0, subs, subs, subs, FALSE, TRUE )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- cublas.context$new( stream )

unit <- cublas.sgemm$new( unit.A.3, unit.B.3, unit.C.3, context = context )
L3   <- cublas.sgemm$new( tens.A.3, tens.B.3, tens.C.3, subs, subs, subs, FALSE, TRUE, context = context )
L0   <- cublas.sgemm$new( tens.A.0, tens.B.0, tens.C.0, subs, subs, subs, FALSE, TRUE )

test <- function(){
  identical( tens.C.3$pull(), tens.C.0$pull() )
}
