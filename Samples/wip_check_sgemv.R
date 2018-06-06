library( cuRious )

# gemv
# y.span(y) = alpha*A.tp(A.span(A)) %*% x.span(x) + beta*y.span(y)
cols <- 10
rows <- 6
subs <- c( 2, 7 )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols*rows) )
vect.y <- as.double( 1:(cols*rows) )

tens.A.3 <- tensor$new( mat.A  , 3 )
tens.x.3 <- tensor$new( vect.x , 3 )
tens.y.3 <- tensor$new( vect.y , 3 )

tens.A.0 <- tensor$new( mat.A  , 0 )
tens.x.0 <- tensor$new( vect.x , 0 )
tens.y.0 <- tensor$new( vect.y , 0 )

handle   <- cublas.handle$new()

L3.sgemv <- cublas.sgemv$new( tens.A.3, tens.x.3, tens.y.3, subs, subs, subs, TRUE, handle = handle )
L0.sgemv <- cublas.sgemv$new( tens.A.0, tens.x.0, tens.y.0, subs, subs, subs, TRUE )

L3.sgemv$run()
L0.sgemv$run()

print( tens.y.3$pull() )
print( tens.y.0$pull() )

clean()
