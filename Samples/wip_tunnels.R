library( cuRious )

src <- tensor$new( c(1) )
dst <- tensor$new( c(1) )

tunnel <- tunnel$new( src, dst )

src$transform( 1L )
