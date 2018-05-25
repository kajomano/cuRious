library( cuRious )
library( R6 )
library( microbenchmark )

fun <- function( var ){
  var + 1
}

test.class <- R6Class(
  "test",
  public = list(
    ptr = NULL,
    ref = function(){
      .Call( "cuR_count_references", self$ptr )
    }
  ),

  active = list(
    set = function( var ){
      print( .Call( "cuR_count_references", var ) )
      self$ptr <- var
    }
  )
)

test <- test.class$new()

var <- 1:3
test$set <- fun( var )

.Call( "cuR_count_references", var )

test$ref()
test$ptr <- fun( 1 )
test$ptr <- 1:3

.Call( "cuR_count_references", 1 )
.Call( "cuR_count_references", 1:3 )
.Call( "cuR_count_references", fun(1) )
