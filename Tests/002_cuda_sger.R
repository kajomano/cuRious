library( cuRious )
library( microbenchmark )

# A = a*x %*% tp(y) + A
# Create matrix tensors and store them in GPU memory
# mult <- 100

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols) )
vect.y <- as.double( 1:(cols) )

tens.A.3 <- tensor$new( mat.A, 3 )
tens.x.3 <- tensor$new( vect.x, 3 )
tens.y.3 <- tensor$new( vect.y, 3 )

tens.A.0 <- tensor$new( mat.A, 0 )
tens.x.0 <- tensor$new( vect.x, 0 )
tens.y.0 <- tensor$new( vect.y, 0 )

unit.A.3 <- tensor$new( 1.0, 3 )
unit.x.3 <- tensor$new( 1.0, 3 )
unit.y.3 <- tensor$new( 1.0, 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- cublas.context$new( stream )

unit <- cublas.sger$new( unit.x.3, unit.y.3, unit.A.3, context = context )
L3   <- cublas.sger$new( tens.x.3, tens.y.3, tens.A.3, subs, subs, subs, context = context )
L0   <- cublas.sger$new( tens.x.0, tens.y.0, tens.A.0, subs, subs, subs )

test <- function(){
  identical( tens.A.3$pull(), tens.A.0$pull() )
}
