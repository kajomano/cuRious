require( cuRious )
require( microbenchmark )

# Create matrix tensors and store them in GPU memory
# GEMM: C = A( m, k ) %*% B( k, n ) + C( m, n )
# mult <- 1

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.B <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
mat.C <- matrix( as.double( 1:(cols*rows) ), ncol = cols )

tens.A.3 <- cuRious::tensor$new( mat.A, 3 )
tens.B.3 <- cuRious::tensor$new( mat.B, 3 )
tens.C.3 <- cuRious::tensor$new( mat.C, 3 )

tens.A.0 <- cuRious::tensor$new( mat.A, 0 )
tens.B.0 <- cuRious::tensor$new( mat.B, 0 )
tens.C.0 <- cuRious::tensor$new( mat.C, 0 )

# Mandatory variables
stream   <- cuRious::stream$new()
context  <- cuRious::cublas.context$new( stream )

L3       <- cuRious::cublas.sgemm$new( tens.A.3, tens.B.3, tens.C.3, subs, subs, subs, FALSE, TRUE, context = context )
L0       <- cuRious::cublas.sgemm$new( tens.A.0, tens.B.0, tens.C.0, subs, subs, subs, FALSE, TRUE )

test <- function( verbose = FALSE ){
  if( verbose ){
    print( tens.C.3$pull() )
    print( tens.C.0$pull() )
  }

  test.thr.equality( tens.C.3$pull(), tens.C.0$pull() )
}

clear <- function(){
  tens.C.3$push( mat.C )
}
