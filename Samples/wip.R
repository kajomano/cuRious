library( cuRious )

a      <- tensor$new( rnorm( 10 ), 0L )
a.copy <- a

b      <- tensor$new( a, 0L, copy = FALSE )

transfer( b, a )
print( a.copy$pull() )
