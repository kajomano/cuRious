library( cuRious )
library( microbenchmark )

rows <- 3000
cols <- 2000
data <- c( 0, 1 )

more.cols <- 4000
more.data <- c( 0, 1 )
subs.cols <- c( 1L, 3L )
subs.cols <- sample( 1:rows, 2000 )

res <- list()

# Level 0-0 ====
# No subsetting ====
x <- matrix( data, rows, cols )
y <- matrix( 0, rows, cols )

copy <- function(){
  .Call( "cuR_copy_obj", x, y, rows*cols )
}

microbenchmark( copy(), times = 10 )
microbenchmark( transfer( x, y ), times = 10 )
print( identical( y, matrix( data, rows, cols ) ) )

# Source subset ====
x <- matrix( more.data, rows, more.cols )
y <- matrix( 0, rows, cols )
microbenchmark( transfer( x, y, subs.cols ), times = 10 )
print( identical( y, matrix( data, rows, more.cols )[ ,subs.cols ] ) )

# Destination subset ====
x <- matrix( data, rows, cols )
y <- matrix( 0, rows, more.cols )
microbenchmark( transfer( x, y, cols.dst = subs.cols ), times = 10 )
print( identical( y[ ,subs.cols ], matrix( data, rows, cols ) ) )

# Destination and source subset ====
x <- matrix( data, rows, more.cols )
y <- matrix( 0, rows, more.cols )
microbenchmark( transfer( x, y, subs.cols, subs.cols ), times = 10 )
print( identical( y[ ,subs.cols ], matrix( data, rows, more.cols )[ ,subs.cols ] ) )

clean.global()
