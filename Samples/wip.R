library( cuRious )
library( microbenchmark )

test <- tensor$new( NULL, 3L )
test$destroy()
test$deploy()
test$ptr
