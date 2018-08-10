library( cuRious )
library( microbenchmark )

# TODO ====
# L0 produces dim match error
# L3 need ot be checked against L0, need to be rewritten for all cublas calls

level <- 3L

tens.A <- cuRious::tensor$new( matrix( as.numeric(1:8),  ncol = 4, nrow = 2 ), level )
tens.B <- cuRious::tensor$new( matrix( as.numeric(1:6),  ncol = 2, nrow = 3 ), level )
tens.C <- cuRious::tensor$new( matrix( as.numeric(1:12), ncol = 3, nrow = 4 ), level )

context <- cuRious::cublas.context$new( level = level )

sgemm   <- cuRious::cublas.sgemm$new( tens.A,
                                      tens.B,
                                      tens.C,
                                      A.tp    = TRUE,
                                      B.tp    = TRUE,
                                      alpha   = 1,
                                      beta    = 0,
                                      context = context )

sgemm$run()

print( tens.C$pull() )
