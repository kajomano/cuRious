library( cuRious )

vect <- as.numeric( 1:36 )

src <- tensor$new( vect )
dst <- tensor$new( src, copy = FALSE )
src$dims <- c( 6, 6 )

tens.span <- tensor.span$new( tens, list( c( 1, 5 ), c( 1, 3 ) ) )



tens.span$span
