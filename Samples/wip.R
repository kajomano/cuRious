library( cuRious )
library( microbenchmark )

vect <- as.numeric( 1:360000 )

src <- tensor$new( vect )
dst <- tensor$new( src, copy = FALSE )
src$dims <- c( 600, 600 )
dst$dims <- c( 600, 600 ) # 9, 4

src.ranged <- tensor.ranged$new( src, list( c( 101, 500 ), c( 101, 300 ) ) )
dst.ranged <- tensor.ranged$new( dst, list( c( 101, 500 ), c( 201, 400 ) ) )

src.perm.1 <- tensor$new( 1:1000 )
src.perm.2 <- tensor$new( 1:1000 )

dst.perm.1 <- tensor$new( 1:1000 )
dst.perm.2 <- tensor$new( 1:1000 )

src.perm.1.ranged <- tensor.ranged$new( src.perm.1, list( c( 1, 400 ) ) )
src.perm.2.ranged <- tensor.ranged$new( src.perm.2, list( c( 1, 200 ) ) )

dst.perm.1.ranged <- tensor.ranged$new( dst.perm.1, list( c( 1, 400 ) ) )
dst.perm.2.ranged <- tensor.ranged$new( dst.perm.2, list( c( 1, 200 ) ) )

src$dims <- c( 360000, 1 )
dst$dims <- c( 360000, 1 )

pip <- pipe$new( src.ranged,
                 dst.ranged,
                 list( src.perm.1.ranged, src.perm.2.ranged ),
                 list( dst.perm.1.ranged, dst.perm.2.ranged ) )

# pip <- pipe$new( src.ranged,
#                  dst.ranged )

# pip$run()
#
# src$clear()
# src$obj.unsafe
# dst$obj.unsafe


print( microbenchmark( pip$run() ) )


# dst$dims <- c( 9, 4 )
# dst$obj.unsafe

# test <- matrix( rnorm( 10*10 ), 10, 10 )
# do.call( `[<-`, list( do.call( `[<-`, list( test, 2:3, 2:3 ) ), , 1 ) )

# test <- matrix( 1:6, 3, 2 )
# do.call( `[`, list( test, 1, T ) )
#
# test[ 2:3 ][ 1 ] <- 0
# test
#
# test <- 1:3
# assign( "test", do.call( `[<-`, list( test, 2, 0 ) ) )
# test
#
# test <- 1:3000
# microbenchmark( { test[ 2000:3000 ] <- 0 } )
# microbenchmark( { assign( "test", do.call( `[<-`, list( test, 2000:3000, 0 ) ) ) } )
#
#
# test <- 1:3
# `[<-`( test, 2, 0 )
# test
#
# tens.dst <- tensor$new( 1:3 )
# tens.src <- tensor$new( 1:3 )
# tens.dst$obj.unsafe <- do.call( `[`, list( tens.src$obj.unsafe ) )
#
# tens.src$clear()
#
#
# do.call( `[<-`, list( test, 2:3,   do.call( `[<-`, list(    do.call( `[`, list( test, 2:3 ) )   , 1, 0 ) )   ) )
# test
#
# microbenchmark( { test[ 2000:3000 ][ 1:500 ] <- 0 } )
#
# test <- list()
# test[[2]] <- 1
#
# for( i in 1:length( test ) ){
#   if( is.null( test[[i]] ) ){
#     test[[i]] <- TRUE
#   }
# }

# test.ranged <- tensor.ranged$new( test, list( c( 1, 50 ), c( 1, 50 ) ) )
#
# obj.rerange = function( wrap, obj = NULL ){
#   ranges <- lapply( 1:nrow( wrap ), function( dim ){
#     if( wrap[ dim, 1 ] != wrap[ dim, 3 ] ){
#       ( wrap[ dim, 2 ] + 1L ):( wrap[ dim, 2 ] + wrap[ dim, 3 ] )
#     }else{
#       TRUE
#     }
#   })
#
#   ranges
# }
#
# microbenchmark( do.call( `[`, c( list( test$obj ), obj.rerange( test.ranged$wrap ) ) ) )


# test2 <- array( 1:6, c( 3, 3 ) )
#
# test2 %*% test1
#
# test2[ 1:2, 1:2 ][ 2:1, 2:1 ] <- 1:4
#
# test1[ , 1 ]
#
# attr( test2, "rank" ) <- 1
