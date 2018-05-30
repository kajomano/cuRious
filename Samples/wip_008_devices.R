library( cuRious )
library( microbenchmark )

tens1 <- tensor$new( rnorm( 10 ), 3L, device = 1 )
tens0 <- tensor$new( tens1, copy = FALSE, device = 0 )

pip <- pipe$new( tens1, tens0 )
pip$run()

tens0$pull()
tens1$pull()

tens1$device <- 0
tens1$pull()

tens0$clear()
cuda.device.set( 1 )
pip$run()
tens1$pull()

tens1$destroy()
tens0$destroy()
