library( cuRious )
library( microbenchmark )

m     <- 6
n     <- 5

mat.A.n       <- matrix( 1, ncol = n, nrow = m )
mat.A.i       <- matrix( 1L, ncol = n, nrow = m )
mat.A.l       <- matrix( TRUE, ncol = n, nrow = m )

mat.A.check.n <- matrix( 0, ncol = n, nrow = m )
mat.A.check.i <- matrix( 0L, ncol = n, nrow = m )
mat.A.check.l <- matrix( FALSE, ncol = n, nrow = m )

for( l in c( 0L, 1L, 2L, 3L ) ){
  tens.A.n <- tensor$new( mat.A.n, l )
  tens.A.i <- tensor$new( mat.A.i, l )
  tens.A.l <- tensor$new( mat.A.l, l )

  clear.obj( tens.A.n )
  clear.obj( tens.A.i )
  clear.obj( tens.A.l )

  print( identical( tens.A.n$pull(), mat.A.check.n ) )
  print( identical( tens.A.i$pull(), mat.A.check.i ) )
  print( identical( tens.A.l$pull(), mat.A.check.l ) )

  destroy.obj( tens.A.n )
  destroy.obj( tens.A.i )
  destroy.obj( tens.A.l )
}

