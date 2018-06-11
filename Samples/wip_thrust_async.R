# Thrust does not seem to do async calls when using some reduction
# print( paste(
#   "sudo",
#   "/usr/local/cuda/bin/nvvp",
#   paste0( R.home(), "/bin/Rscript" ),
#   paste0( getwd(), "/Samples/wip_thrust_async.R" )
# ))

library( cuRious )
library( microbenchmark )

tens3      <- tensor$new( matrix( as.double( 1:10^6 ), 1000, 1000 ), 3L )
tens3.vect <- tensor$new( NULL, 3L, c( 1L, 1000L ), "i" )

stream <- cuda.stream$new( TRUE )

thrust3 <- thrust.cmin.pos$new( tens3, tens3.vect, stream = stream )
thrust3$run()

# for( i in 1:10 ){
#   thrust3$run()
# }

# stream$sync()

# microbenchmark( thrust3$run() )
