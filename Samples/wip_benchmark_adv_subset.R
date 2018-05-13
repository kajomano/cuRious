library( cuRious )
library( microbenchmark )

# TODO ====
# Definitely core-bound, need to use mutliple threads/blocks, and use
# cudaMemcpy wherever possible

cols <- 10^3

tens.X.1 <- tensor$new( NULL, 1L, c( 1000, cols ), "n" )
tens.Y.1 <- tensor$new( NULL, 1L, c( 1000, cols ), "n" )

tens.X.3 <- tensor$new( NULL, 3L, c( 1000, cols ), "n" )
tens.Y.3 <- tensor$new( NULL, 3L, c( 1000, cols ), "n" )

tens.X.perm.1 <- tensor$new( as.integer( 1:cols ), 1L )
tens.Y.perm.1 <- tensor$new( as.integer( 1:cols ), 1L )

tens.X.perm.3 <- tensor$new( as.integer( 1:cols ), 3L )
tens.Y.perm.3 <- tensor$new( as.integer( 1:cols ), 3L )

transfer.1 <- function(){
  transfer( tens.X.1,
            tens.Y.1,
            tens.X.perm.1,
            tens.Y.perm.1 )
}

transfer.3 <- function(){
  transfer( tens.X.3,
            tens.Y.3,
            tens.X.perm.3,
            tens.Y.perm.3 )
}

microbenchmark( transfer.1(), times = 100 )
microbenchmark( transfer.3(), times = 100 )

# clean()
