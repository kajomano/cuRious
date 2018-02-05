library( cuRious )
library( microbenchmark )

# TODO ====
# Templatelt transfer callok

# TODO ====
# Legyen forceolt destructor a tensorokban ha felj0nnek

# TODO ====
# Subsettelt async memcpy L3-L3

library( cuRious )
library( microbenchmark )

for( j in 1:100 ){
  print( paste0( "Iter: ", j ) )

  for( t in 1:3 ){

    print( paste0( "Type: ", t ) )

    rows      <- round( runif( 1, 1, 3000 ) )
    cols      <- round( runif( 1, 1, 2000 ) )
    sub.cols  <- round( runif( 1, 1, cols ) )

    data      <-switch(
      t,
      sample( c( TRUE, FALSE ), cols*rows, TRUE ),
      sample( c( 1L, 2L ), cols*rows, TRUE ),
      sample( c( 1, 2 ), cols*rows, TRUE )
    )

    lev.comb <- list()
    for( lev.x in c(0,1,2) ){
      for( lev.y in c(0,1,2) ){
        lev.comb <- c( lev.comb, list( list(lev.x = lev.x, lev.y = lev.y) ) )
      }
    }

    for( i in 1:length( lev.comb ) ){
      lev.x <- lev.comb[[i]]$lev.x
      lev.y <- lev.comb[[i]]$lev.y

      print( paste0( lev.x, "->", lev.y ) )

      # No subsetting ====
      print( "No subsetting" )
      x <- tensor$new( matrix( data, rows, cols ) )$transform( lev.x )
      y <- tensor$new( matrix( data[[1]], rows, cols ) )$transform( lev.y )
      transfer( x, y )
      if( !identical( y$pull(), x$pull() ) ){
        stop("Error in no subsetting")
      }

      # Range subsetting ====
      print( "Range subsetting" )
      x <- tensor$new( matrix( data, rows, cols ) )$transform( lev.x )
      y <- tensor$new( matrix( data[[1]], rows, cols ) )$transform( lev.y )
      sub.x <- list( 1, sub.cols )
      sub.y <- list( cols - sub.cols + 1, cols )
      transfer( x, y, sub.x, sub.y )
      if( !identical( x$pull()[, sub.x[[1]]:sub.x[[2]]], y$pull()[, sub.y[[1]]:sub.y[[2]]] ) ){
        stop("Error in range subsetting")
      }

      # Individual subsetting ====
      print( "Individual subsetting" )
      x <- tensor$new( matrix( data, rows, cols ) )$transform( lev.x )
      y <- tensor$new( matrix( data[[1]], rows, cols ) )$transform( lev.y )
      sub.x <- sample( 1:cols, sub.cols )
      sub.y <- sample( 1:cols, sub.cols )
      transfer( x, y, sub.x, sub.y )
      if( !identical( x$pull()[, sub.x], y$pull()[, sub.y] ) ){
        stop("Error in individual subsetting")
      }
    }

    lev.comb <- list()
    for( lev.x in c(3) ){
      for( lev.y in c(0,1,2,3) ){
        lev.comb <- c( lev.comb, list( list(lev.x = lev.x, lev.y = lev.y) ) )
      }
    }
    for( lev.x in c(0,1,2) ){
      for( lev.y in c(3) ){
        lev.comb <- c( lev.comb, list( list(lev.x = lev.x, lev.y = lev.y) ) )
      }
    }

    for( lev.x in c(3) ){
      for( lev.y in c(3) ){
        lev.comb <- c( lev.comb, list( list(lev.x = lev.x, lev.y = lev.y) ) )
      }
    }

    stream <- cuda.stream$new()
    stream$activate()

    for( i in 1:length( lev.comb ) ){
      lev.x <- lev.comb[[i]]$lev.x
      lev.y <- lev.comb[[i]]$lev.y

      print( paste0( lev.x, "->", lev.y ) )

      # No subsetting ====
      print( "No subsetting L3" )
      x <- tensor$new( matrix( data, rows, cols ) )$transform( lev.x )
      y <- tensor$new( matrix( data[[1]], rows, cols ) )$transform( lev.y )
      transfer( x, y )
      if( !identical( x$pull(), y$pull() ) ){
        stop("Error in no subsetting L3")
      }

      # Range subsetting ====
      print( "Range subsetting L3" )
      x <- tensor$new( matrix( data, rows, cols ) )$transform( lev.x )
      y <- tensor$new( matrix( data[[1]], rows, cols ) )$transform( lev.y )
      sub.x <- list( 1, sub.cols )
      sub.y <- list( cols - sub.cols + 1, cols )
      transfer( x, y, sub.x, sub.y )
      if( !identical( x$pull()[, sub.x[[1]]:sub.x[[2]]], y$pull()[, sub.y[[1]]:sub.y[[2]]] ) ){
        stop("Error in range subsetting L3")
      }
    }
  }
  clean.global()
}
