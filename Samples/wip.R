library( cuRious )

vect <- as.numeric( 1:36 )

src <- tensor$new( vect )
dst <- tensor$new( src, copy = FALSE )
src$dims <- c( 6, 6 )
dst$dims <- c( 9, 4 )

src.span <- tensor.span$new( src, list( c( 1, 5 ), c( 1, 3 ) ) )
dst.span <- tensor.span$new( dst, list( c( 1, 5 ), c( 2, 4 ) ) )

src.perm.1 <- tensor$new( 1:10 )
src.perm.2 <- tensor$new( 1:10 )

dst.perm.1 <- tensor$new( 1:10 )
dst.perm.2 <- tensor$new( 1:10 )

src.perm.1.span <- tensor.span$new( src.perm.1, list( c( 1, 5 ) ) )
src.perm.2.span <- tensor.span$new( src.perm.2, list( c( 1, 3 ) ) )

dst.perm.1.span <- tensor.span$new( dst.perm.1, list( c( 1, 5 ) ) )
dst.perm.2.span <- tensor.span$new( dst.perm.2, list( c( 1, 3 ) ) )

pip <- pipe$new( src.span,
                 dst.span,
                 list( src.perm.1.span, src.perm.2.span ),
                 list( dst.perm.1.span, dst.perm.2.span ) )

pip$run()
