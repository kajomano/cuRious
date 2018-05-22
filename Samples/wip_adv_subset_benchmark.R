library( cuRious )
library( microbenchmark )

cols <- 10^3

tens.X.1 <- tensor$new( NULL, 1L, c( 1000, cols ), "n" )
tens.Y.1 <- tensor$new( NULL, 1L, c( 1000, cols ), "n" )

tens.X.3 <- tensor$new( NULL, 3L, c( 1000, cols ), "n" )
tens.Y.3 <- tensor$new( NULL, 3L, c( 1000, cols ), "n" )

tens.X.perm.1 <- tensor$new( as.integer( 1:cols ), 1L )
tens.Y.perm.1 <- tensor$new( as.integer( 1:cols ), 1L )

tens.X.perm.3 <- tensor$new( as.integer( 1:cols ), 3L )
tens.Y.perm.3 <- tensor$new( as.integer( 1:cols ), 3L )

stream <- cuda.stream$new()

pipe.1.nosub <- pipe$new( tens.X.1,
                          tens.Y.1 )

pipe.3.bothsub <- pipe$new( tens.X.3,
                            tens.Y.3,
                            tens.X.perm.3,
                            tens.Y.perm.3 )

pipe.3.nosub <- pipe$new( tens.X.3,
                          tens.Y.3 )

pipe.3.srcsub <- pipe$new( tens.X.3,
                           tens.Y.3,
                           tens.X.perm.3 )

pipe.3.dstsub <- pipe$new( tens.X.3,
                           tens.Y.3,
                           NULL,
                           tens.Y.perm.3 )

pipe.3.bothsub.async <- pipe$new( tens.X.3,
                                  tens.Y.3,
                                  tens.X.perm.3,
                                  tens.Y.perm.3,
                                  stream = stream )

pipe.3.nosub.async <- pipe$new( tens.X.3,
                                tens.Y.3,
                                stream = stream )

pipe.3.srcsub.async <- pipe$new( tens.X.3,
                                 tens.Y.3,
                                 tens.X.perm.3,
                                 stream = stream )

pipe.3.dstsub.async <- pipe$new( tens.X.3,
                                 tens.Y.3,
                                 NULL,
                                 tens.Y.perm.3,
                                 stream = stream )

# ----------------------------------------------------------------------

times <- 100

print( microbenchmark( pipe.1.nosub$run(), times = times ) )

print( microbenchmark( pipe.3.bothsub$run(), times = times ) )
print( microbenchmark( pipe.3.nosub$run(),   times = times ) )
print( microbenchmark( pipe.3.srcsub$run(),  times = times ) )
print( microbenchmark( pipe.3.dstsub$run(),  times = times ) )

print( microbenchmark( pipe.3.bothsub.async$run(), times = times ) )
print( microbenchmark( pipe.3.nosub.async$run(),   times = times ) )
print( microbenchmark( pipe.3.srcsub.async$run(),  times = times ) )
print( microbenchmark( pipe.3.dstsub.async$run(),  times = times ) )

clean()
