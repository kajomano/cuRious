library( cuRious )
library( microbenchmark )

rows <- 30
cols <- 20
data <- round( rnorm( cols*rows ) )

more.cols <- 40
more.data <- round( rnorm( more.cols*rows ) )
subs.cols <- sample( 1:more.cols, cols )

lev.x <- 0
lev.y <- 0

# No subsetting ====
x <- tensor$new( matrix( data, rows, cols ) )
y <- tensor$new( matrix( 0, rows, cols ) )
x$transform( lev.x )
y$transform( lev.y )

# microbenchmark( transfer( x, y ), times = 10 )
transfer( x, y )
res <- identical( y$pull(), matrix( data, rows, cols ) )
if( !res ) stop("No subsetting")

# Source subset ====
x <- tensor$new( matrix( more.data, rows, more.cols ) )
y <- tensor$new( matrix( 0, rows, cols ) )
x$transform( lev.x )
y$transform( lev.y )

# microbenchmark( transfer( x, y, subs.cols ), times = 10 )
transfer( x, y, subs.cols )
res <- identical( y$pull(), matrix( more.data, rows, more.cols )[ ,subs.cols ] )
if( !res ) stop("Source subset")

# Destination subset ====
x <- tensor$new( matrix( data, rows, cols ) )
y <- tensor$new( matrix( 0, rows, more.cols ) )
x$transform( lev.x )
y$transform( lev.y )

# microbenchmark( transfer( x, y, cols.dst = subs.cols ), times = 10 )
transfer( x, y, cols.dst = subs.cols )
res <- identical( y$pull()[ ,subs.cols ], matrix( data, rows, cols ) )
if( !res ) stop("Destination subset")

# Destination and source subset ====
x <- tensor$new( matrix( data, rows, more.cols ) )
y <- tensor$new( matrix( 0, rows, more.cols ) )
x$transform( lev.x )
y$transform( lev.y )

# microbenchmark( transfer( x, y, subs.cols, subs.cols ), times = 10 )
transfer( x, y, subs.cols, subs.cols )
res <- identical( y$pull()[ ,subs.cols ], matrix( data, rows, more.cols )[ ,subs.cols ] )
if( !res ) stop("Destination and source subset")

clean.global()
