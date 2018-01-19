library( cuRious )
library( microbenchmark )

rows <- 3000
cols <- 2000
data <- c( 0, 1 )
more.rows <- 5
subs.rows <- c( 1L, 2L, 4L )

res <- list()

# Level 0-0
# No subsetting
x <- matrix( data, rows, cols )
y <- matrix( 0, rows, cols )
transfer( x, y )
res <- c( res, identical( y, matrix( data, rows, cols ) ) )

# Source subset
x <- matrix( data, more.rows, cols )
y <- matrix( 0, rows, cols )
transfer( x, y, subs.rows )
# ITT
res <- c( res, identical( y, matrix( data, more.rows, cols )[ subs.rows, ] ) )
