library( cuRious )
library( microbenchmark )

cols.l <- 1000
rows.l <- 1000

cols.s <- 900
rows.s <- 900

mat.l  <- matrix( rnorm( cols.l*rows.l ), ncol = cols.l )
mat.s  <- matrix( rnorm( cols.s*rows.s ), ncol = cols.s )

tens.mat.l <- tensor$new( mat.l, 3 )
tens.mat.s <- tensor$new( mat.s, 3 )

tens.vect.l <- tensor$new( NULL, 3, dims = c( 1, cols.l ), "i" )
tens.vect.s <- tensor$new( NULL, 3, dims = c( 1, cols.s ), "i" )

stream  <- cuda.stream$new( TRUE )
context <- thrust.context$new( stream )

cmin.l <- thrust.cmin.pos$new( tens.mat.l, tens.vect.l, context = context )
cmin.s <- thrust.cmin.pos$new( tens.mat.s, tens.vect.s, context = context )

microbenchmark( cmin.l$run(), times = 10 )
microbenchmark( cmin.s$run(), times = 1 )

clean()
