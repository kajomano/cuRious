library( cuRious )

# TODO ====
# Check the severing mechanism

vector <- c( 1, 2 )

tens.a <- tensor$new( vector )
tens.b <- tensor$new( vector, copy = FALSE )

fusi.a <- thrust.pow$new( tens.b, tens.a )

fusi.a$run()

vector
