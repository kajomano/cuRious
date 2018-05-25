library( microbenchmark )
library( cuRious )
library( R6 )

test.class <- R6Class(
  "test",
  public = list(
    data = NULL,
    fun  = function(){
      self$data[[1]] <- 1

      invisible( TRUE )
    }
  )
)

mat <- matrix( 0, 1000, 1000 )
test <- test.class$new()
test$data <- mat
tracemem( mat )

test$fun()

res <- microbenchmark( test$fun(), times = 10 )
res$time

clean()
