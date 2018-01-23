library( cuRious )
library( microbenchmark )

big.n <- 10^7
tens.Y <- tensor$new( rep(0, times = big.n ) )
tens.X <- tensor$new( rep(1, times = big.n ) )
tens.Y$transform( 1 )

microbenchmark( transfer( tens.X, tens.Y, threads = 4L ), times = 100 )

clean.global()


identical( tens.Y$pull(), tens.X$pull() )


for( j in 1:100000 ){

  rows      <- runif( 1, 1, 3000 )
  cols      <- runif( 1, 1, 2000 )
  more.cols <- runif( 1, cols, 4000 )
  threads   <- as.integer( runif( 1, 1, 4 ) )

  data      <- round( rnorm( cols*rows ) )
  more.data <- round( rnorm( more.cols*rows ) )
  subs.cols <- sample( 1:more.cols, cols )

  lev.comb <- list()
  for( lev.x in c(0,1,2) ){
    for( lev.y in c(0,1,2) ){
      lev.comb <- c( lev.comb, list( list(lev.x = lev.x, lev.y = lev.y) ) )
    }
  }
  lev.comb <- c( lev.comb, list( list(lev.x = 0, lev.y = 3) ) )
  lev.comb <- c( lev.comb, list( list(lev.x = 3, lev.y = 0) ) )

  for( i in 1:length( lev.comb ) ){
    lev.x <- lev.comb[[i]]$lev.x
    lev.y <- lev.comb[[i]]$lev.y

    # No subsetting ====
    x <- tensor$new( matrix( data, rows, cols ) )
    y <- tensor$new( matrix( 0, rows, cols ) )
    x$transform( lev.x )
    y$transform( lev.y )

    # microbenchmark( transfer( x, y ), times = 10 )
    transfer( x, y, threads = threads )
    res <- identical( y$pull(), matrix( data, rows, cols ) )
    if( !res ) stop("No subsetting")

    # Source subset ====
    x <- tensor$new( matrix( more.data, rows, more.cols ) )
    y <- tensor$new( matrix( 0, rows, cols ) )
    x$transform( lev.x )
    y$transform( lev.y )

    # microbenchmark( transfer( x, y, subs.cols ), times = 10 )
    transfer( x, y, subs.cols, threads = threads )
    res <- identical( y$pull(), matrix( more.data, rows, more.cols )[ ,subs.cols ] )
    if( !res ) stop("Source subset")

    # Destination subset ====
    x <- tensor$new( matrix( data, rows, cols ) )
    y <- tensor$new( matrix( 0, rows, more.cols ) )
    x$transform( lev.x )
    y$transform( lev.y )

    # microbenchmark( transfer( x, y, cols.dst = subs.cols ), times = 10 )
    transfer( x, y, cols.dst = subs.cols, threads = threads )
    res <- identical( y$pull()[ ,subs.cols ], matrix( data, rows, cols ) )
    if( !res ) stop("Destination subset")

    # Destination and source subset ====
    x <- tensor$new( matrix( more.data, rows, more.cols ) )
    y <- tensor$new( matrix( 0, rows, more.cols ) )
    x$transform( lev.x )
    y$transform( lev.y )

    # microbenchmark( transfer( x, y, subs.cols, subs.cols ), times = 10 )
    transfer( x, y, subs.cols, subs.cols, threads = threads )
    res <- identical( y$pull()[ ,subs.cols ], matrix( more.data, rows, more.cols )[ ,subs.cols ] )
    if( !res ) stop("Destination and source subset")
  }

  lev.comb <- list()
  lev.comb <- c( lev.comb, list( list(lev.x = 3, lev.y = 3) ) )
  lev.comb <- c( lev.comb, list( list(lev.x = 2, lev.y = 3) ) )
  lev.comb <- c( lev.comb, list( list(lev.x = 3, lev.y = 2) ) )

  stream <- cuda.stream$new()
  stream$activate()

  for( i in 1:length( lev.comb ) ){
    lev.x <- lev.comb[[i]]$lev.x
    lev.y <- lev.comb[[i]]$lev.y

    # No subsetting ====
    x <- tensor$new( matrix( data, rows, cols ) )
    y <- tensor$new( matrix( 0, rows, cols ) )
    x$transform( lev.x )
    y$transform( lev.y )

    microbenchmark( transfer( x, y, stream = stream ), times = 10 )
    microbenchmark( transfer( x, y ), times = 10 )
    transfer( x, y, stream = stream )
    res <- identical( y$pull(), matrix( data, rows, cols ) )
    if( !res ) stop("No subsetting async")
  }

  print( j )
  clean.global()

}
