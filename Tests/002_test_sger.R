library( cuRious )
library( microbenchmark )

# A = a*x %*% tp(y) + A
# Create matrix tensors and store them in GPU memory
mult <- 100

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

stream1 <- stream$new()
context <- cublas.context$new( stream1 )

L3.sger <- cublas.sger$new( tens.x.3, tens.y.3, tens.A.3, subs, subs, subs, context = context )
L0.sger <- cublas.sger$new( tens.x.0, tens.y.0, tens.A.0, subs, subs, subs )

microbenchmark( L3.sger$run() )
microbenchmark( L0.sger$run() )

# L3.sger$run()
# L0.sger$run()
#
# print( tens.A.3$pull() )
# print( tens.A.0$pull() )

clean()
