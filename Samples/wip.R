library( cuRious )
library( microbenchmark )
library( R6 )

obj1 <- 1.0
tracemem( obj1 )
obj2 <- 2.0

tens1 <- tensor$new( obj1, 0L )
tens1$ptr

tens1$sever()
.Call( "cuR_transfer", obj2, tens1$ptr, 0L, 0L, "n", tens1$dims, tens1$dims, tens1$dims, NULL, NULL, NULL, NULL, NULL )

obj1

clean()

# dims <- c( 3000L, 2000L )
# type <- "n"
#
# obj1 <- matrix( rnorm( prod( dims ) ), dims[[1]], dims[[2]] )
# obj2 <- obj.create( dims, type )
#
# tens.L2   <- .Call( "cuR_tensor_create", 2L, dims, type )
# tens.L3.1 <- .Call( "cuR_tensor_create", 3L, dims, type )
# tens.L3.2 <- .Call( "cuR_tensor_create", 3L, dims, type )
#
# # .Call( "cuR_tensor_clear", obj1, 0L, dims, type )
#
# .Call( "cuR_transfer", obj1, tens.L2, 0L, 2L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
# .Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
#
# stream <- cuda.stream$new()
#
# L2.transfer <- function(){
#   .Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
# }
# L2.transfer.async <- function(){
#   .Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, stream$ptr )
# }
#
# print( microbenchmark( L2.transfer() ) )
# print( microbenchmark( L2.transfer.async() ) )
#
# L3.transfer <- function(){
#   .Call( "cuR_transfer", tens.L3.1, tens.L3.2, 3L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
# }
# L3.transfer.async <- function(){
#   .Call( "cuR_transfer", tens.L3.1, tens.L3.2, 3L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, stream$ptr )
# }
#
# L3.transfer()
#
# print( microbenchmark( L3.transfer() ) )
# print( microbenchmark( L3.transfer.async() ) )
#
# .Call( "cuR_transfer", tens.L3.2, tens.L2, 3L, 2L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
# .Call( "cuR_transfer", tens.L2, obj2, 2L, 0L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
#
# print( obj2 )
#
# .Call( "cuR_tensor_destroy", tens.L2, 2L, type )
# .Call( "cuR_tensor_destroy", tens.L3.1, 3L, type )
# .Call( "cuR_tensor_destroy", tens.L3.2, 3L, type )
#
# clean()
