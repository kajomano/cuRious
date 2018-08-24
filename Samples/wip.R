library( cuRious )

vect <- as.numeric( 1:36 )

src <- tensor$new( vect )
dst <- tensor$new( src, copy = FALSE )
src$dims <- c( 6, 6 )
dst$dims <- c( 9, 4 )

src.ranged <- tensor.ranged$new( src, list( c( 1, 5 ), c( 1, 3 ) ) )
dst.ranged <- tensor.ranged$new( dst, list( c( 1, 5 ), c( 2, 4 ) ) )

src.perm.1 <- tensor$new( 1:10 )
src.perm.2 <- tensor$new( 1:10 )

dst.perm.1 <- tensor$new( 1:10 )
dst.perm.2 <- tensor$new( 1:10 )

src.perm.1.ranged <- tensor.ranged$new( src.perm.1, list( c( 1, 5 ) ) )
src.perm.2.ranged <- tensor.ranged$new( src.perm.2, list( c( 1, 3 ) ) )

dst.perm.1.ranged <- tensor.ranged$new( dst.perm.1, list( c( 1, 5 ) ) )
dst.perm.2.ranged <- tensor.ranged$new( dst.perm.2, list( c( 1, 3 ) ) )

pip <- pipe$new( src.ranged,
                 dst.ranged,
                 list( src.perm.1.ranged, src.perm.2.ranged ),
                 list( dst.perm.1.ranged, dst.perm.2.ranged ) )

pip$run()

# test1 <- array( 1:3 )
# test2 <- array( 1:6, c( 3, 3 ) )
#
# test2 %*% test1
#
# test2[ 1:2, 1:2 ][ 2:1, 2:1 ] <- 1:4
#
# test1[ , 1 ]
#
# attr( test2, "rank" ) <- 1
