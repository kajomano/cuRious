library( cuRious )
library( microbenchmark )

# Pow ====

tens.test   <- tensor$new( matrix( as.double( 1:9 ), 3, 3 ), 3L )
thrust.test <- thrust.pow$new( tens.test, tens.test, pow = 2 )

thrust.test$run()

print( tens.test$pull() )

tens0 <- tensor$new( matrix( as.double( 1:10^6 ), 1000, 1000 ), 0L )
tens3 <- tensor$new( tens0, 3L )

tens0.res <- tensor$new( tens0 )
tens3.res <- tensor$new( tens3 )

stream <- cuda.stream$new( FALSE )
allocator  <- thrust.allocator$new()

thrust0 <- thrust.pow$new( tens0, tens0.res )
thrust3 <- thrust.pow$new( tens3, tens3.res, allocator = allocator, stream = stream )

print( microbenchmark( thrust0$run() ) )
print( microbenchmark( thrust3$run() ) )
stream$activate()
print( microbenchmark( thrust3$run() ) )
stream$deactivate()

# Cmin pos ====
tens.vect <- tensor$new( NULL, 3L, c( 1L, 3L ), "i" )
thrust.test <- thrust.cmin.pos$new( tens.test, tens.vect )

thrust.test$run()

print( tens.vect$pull() )

tens0.vect <- tensor$new( NULL, 0L, c( 1L, 1000L ), "i" )
tens3.vect <- tensor$new( NULL, 3L, c( 1L, 1000L ), "i" )

thrust0 <- thrust.cmin.pos$new( tens0, tens0.vect )
thrust3 <- thrust.cmin.pos$new( tens3, tens3.vect, stream = stream )

# ITT ====
# Nem async a streames call

print( microbenchmark( thrust0$run() ) )
print( microbenchmark( thrust3$run() ) )
stream$activate()
print( microbenchmark( thrust3$run() ) )
stream$deactivate()
