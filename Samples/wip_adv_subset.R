tens.X <- tensor$new( matrix( as.numeric( 1:6 ), 2, 3 ), 0L )
tens.Y <- tensor$new( matrix( as.numeric( 0   ), 2, 3 ), 1L )

tens.X.perm <- tensor$new( c(1L, 3L, 2L) )
tens.Y.perm <- tensor$new( c(3L, 2L, 1L) )

transfer( tens.X,
          tens.Y,
          tens.X.perm,
          tens.Y.perm,
          c( 1L, 2L ),
          c( 2L, 3L ) )

print( tens.X$pull() )
print( tens.Y$pull() )
