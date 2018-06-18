library( cuRious )
library( microbenchmark )

# gemv
# y.span(y) = alpha*A.tp(A.span(A)) %*% x.span(x) + beta*y.span(y)
mult <- 100

cols <- 10 * mult
rows <- 6 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols) )
vect.y <- as.double( 1:(cols) )

tens.A.3 <- tensor$new( mat.A  , 3 )
tens.x.3 <- tensor$new( vect.x , 3 )
tens.y.3 <- tensor$new( vect.y , 3 )

tens.A.0 <- tensor$new( mat.A  , 0 )
tens.x.0 <- tensor$new( vect.x , 0 )
tens.y.0 <- tensor$new( vect.y , 0 )

stream1 <- stream$new()
context <- cublas.context$new( stream1 )

L3.sgemv <- cublas.sgemv$new( tens.A.3, tens.x.3, tens.y.3, subs, subs, subs, TRUE, context = context )
L0.sgemv <- cublas.sgemv$new( tens.A.0, tens.x.0, tens.y.0, subs, subs, subs, TRUE )

# TODO ====
# Check this, this kills the cublas execution
stream1$destroy()

microbenchmark( L3.sgemv$run() )
microbenchmark( L0.sgemv$run() )

# L3.sgemv$run()
# L0.sgemv$run()
#
# print( tens.y.3$pull() )
# print( tens.y.0$pull() )

clean()
