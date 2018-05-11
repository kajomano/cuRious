library( cuRious )

# A = a*x %*% tp(y) + A

# Create matrix tensors and store them in GPU memory
cols <- 10
rows <- 6
subs <- c( 2, 7 )

mat.A  <- matrix( as.double( 1:(cols*rows) ), ncol = cols )
vect.x <- as.double( 1:(cols) )
vect.y <- as.double( 1:(cols) )

tens.A.3 <- tensor$new( mat.A , 3 )
tens.x.3 <- tensor$new( vect.x , 3 )
tens.y.3 <- tensor$new( vect.y , 3 )

tens.A.0 <- tensor$new( mat.A , 0 )
tens.x.0 <- tensor$new( vect.x , 0 )
tens.y.0 <- tensor$new( vect.y , 0 )

handle <- cublas.handle$new()

# cublas.sger( tens.x.3, tens.y.3, tens.A.3, subs, subs, subs, handle = handle )
cublas.sger( tens.x.0, tens.y.0, tens.A.0, subs, subs, subs )

print( tens.A.3$pull() )
print( tens.A.0$pull() )

clean()
