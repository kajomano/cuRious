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

transfer.1 <- function(){
  transfer( tens.X.1,
            tens.Y.1,
            tens.X.perm.1,
            tens.Y.perm.1 )
}

transfer.3.bothsub <- function(){
  transfer( tens.X.3,
            tens.Y.3,
            tens.X.perm.3,
            tens.Y.perm.3 )
}

transfer.3.nosub <- function(){
  transfer( tens.X.3,
            tens.Y.3 )
}

transfer.3.srcsub <- function(){
  transfer( tens.X.3,
            tens.Y.3,
            tens.X.perm.3 )
}

transfer.3.dstsub <- function(){
  transfer( tens.X.3,
            tens.Y.3,
            NULL,
            tens.Y.perm.3 )
}

# ----------------------------------------------------------------------

times <- 10

print( microbenchmark( transfer.1(), times = times ) )

print( microbenchmark( transfer.3.bothsub(), times = times ) )
print( microbenchmark( transfer.3.nosub(),   times = times ) )
print( microbenchmark( transfer.3.srcsub(),  times = times ) )
print( microbenchmark( transfer.3.dstsub(),  times = times ) )

clean()
