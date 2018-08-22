library( cuRious )

vect <- as.numeric( 1:36 )

src <- tensor$new( vect )
dst <- tensor$new( src, copy = FALSE )
src$dims <- c( 6, 6 )
dst$dims <- c( 9, 4 )

src.span <- tensor.span$new( src, list( c( 1, 5 ), c( 1, 3 ) ) )
dst.span <- tensor.span$new( dst, list( c( 1, 5 ), c( 2, 4 ) ) )

pipe$new( src.span, dst.span )
