library( cuRious )
library( microbenchmark )

stream  <- cuRious::stream$new( NULL, 0 )
context <- cuRious::pipe.context$new( 4, stream )
