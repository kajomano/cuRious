library( cuRious )
library( microbenchmark )

# TODO ====
# L0 produces dim match error
# L3 need ot be checked against L0, need to be rewritten for all cublas calls

level <- 3L

tens.A <- cuRious::tensor$new( matrix( as.numeric(1:6), ncol = 3, nrow = 2 ), level )
tens.x <- cuRious::tensor$new( as.numeric(1:2), level )
tens.y <- cuRious::tensor$new( as.numeric(1:3), level )

context <- cuRious::cublas.context$new( level = level )

sgemv   <- cuRious::cublas.sgemv$new( tens.A,
                                      tens.x,
                                      tens.y,
                                      A.tp    = TRUE,
                                      alpha   = 1,
                                      beta    = 0,
                                      context = context )

sgemv$run()

print( tens.y$pull() )
