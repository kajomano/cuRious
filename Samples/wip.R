library( cuRious )
library( microbenchmark )
library( R6 )

clean()

dims <- c( 3000L, 2000L )
type <- "n"

obj1 <- matrix( rnorm( prod( dims ) ), dims[[1]], dims[[2]] )
obj2 <- obj.create( dims, type )

tens.L2   <- .Call( "cuR_tensor_create", 2L, dims, type )
tens.L3.1 <- .Call( "cuR_tensor_create", 3L, dims, type )
tens.L3.2 <- .Call( "cuR_tensor_create", 3L, dims, type )

# .Call( "cuR_tensor_clear", obj1, 0L, dims, type )

.Call( "cuR_transfer", obj1, tens.L2, 0L, 2L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
.Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )

stream <- cuda.stream$new()

L2.transfer <- function(){
  .Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
}
L2.transfer.async <- function(){
  .Call( "cuR_transfer", tens.L2, tens.L3.1, 2L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, stream$ptr )
}

print( microbenchmark( L2.transfer() ) )
print( microbenchmark( L2.transfer.async() ) )

L3.transfer <- function(){
  .Call( "cuR_transfer", tens.L3.1, tens.L3.2, 3L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
}
L3.transfer.async <- function(){
  .Call( "cuR_transfer", tens.L3.1, tens.L3.2, 3L, 3L, type, dims, dims, dims, NULL, NULL, NULL, NULL, stream$ptr )
}

print( microbenchmark( L3.transfer() ) )
print( microbenchmark( L3.transfer.async() ) )

.Call( "cuR_transfer", tens.L3.2, tens.L2, 3L, 2L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )
.Call( "cuR_transfer", tens.L2, obj2, 2L, 0L, type, dims, dims, dims, NULL, NULL, NULL, NULL, NULL )

print( obj2 )

.Call( "cuR_tensor_destroy", tens.L2, 2L, type )
.Call( "cuR_tensor_destroy", tens.L3.1, 3L, type )
.Call( "cuR_tensor_destroy", tens.L3.2, 3L, type )

clean()

# test.class <- R6Class(
#   "test",
#   private = list(
#     ptr = 1
#   )
# )
#
# test <- test.class$new()
#
# microbenchmark( test[['.__enclos_env__']]$private$ptr )
#
# test$.__enclos_env__

# TODO ====
# Own reference counting
# get_private <- function(x) {
#   x[['.__enclos_env__']]$private
# }



# testenv <- new.env()
#
# testenv$obj1 <- 1:100
# .Internal( inspect( testenv$obj1 ) )
# tracemem( testenv$obj1 )
#
# obj2 <- obj1
#
# .Call( "cuR_reset_named", testenv$obj1 )
#
# # Operation
# .Call( "cuR_access_ptr", testenv )
# .Internal( inspect(testenv$obj1) )
#






obj2 <- 1:10



obj2 <- .Call( "cuR_shallow_duplicate", obj1 )
obj2 <- .Call( "cuR_shallow_duplicate", obj2 )
obj1[[1]] <- 1

.Internal( inspect(obj2) )
.Internal( inspect(1:10) )


obj2 <- obj1
.Call( "cuR_get_obj_ptr", obj2 )

tracemem( obj2 )

obj3 <- obj2

microbenchmark( obj2[[1]] <- obj2[[1]] )
obj2[[1]] <- obj2[[1]]
.Call( "cuR_get_obj_ptr", obj2 )

.Internal( inspect(obj2) )

# TODO =====
# Write own severing function

# obj1 <- matrix( rnorm(1000) )
# tens1 <- tensor$new( obj1 )
#
# .Call( "cuR_get_obj_ptr", tens1$ptr )
# .Call( "cuR_get_obj_ptr", obj1 )
#
# tens2 <- tensor$new( tens1 )
#
# .Call( "cuR_get_obj_ptr", tens2$ptr )
#
# # transfer( tens1, tens2 )
# #
# # .Call( "cuR_get_obj_ptr", tens2$ptr )
# # .Call( "cuR_get_obj_ptr", tens1$ptr )
# # .Call( "cuR_get_obj_ptr", obj1 )
#
# pip <- pipe$new( tens1, tens2 )
# pip$run()
# pip$run()
#
# microbenchmark( pip$run() )

# Something's not right, I think ptr-s are refreshed each time

# TODO ====
# Test that truly, no unintended side-effects occur

# TODO ====
# Test that transfers actually complete
# NOPE

# test1 <- 1
# test2 <- 1
#
# test_ptr1 <- .Call( "cuR_get_obj_ptr", test1 )
# test_ptr2 <- .Call( "cuR_get_obj_ptr", test2 )
#
# .Call( "cuR_compare_obj_ptr", test_ptr1, test_ptr2 )
# microbenchmark( .Call( "cuR_compare_obj_ptr", test_ptr1, test_ptr2 ) )
