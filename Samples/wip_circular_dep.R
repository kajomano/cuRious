library( R6 )

test <- R6Class(
  "test",
  public = list(
    a = NULL,
    contain = function(){
      self$a <- test$new()
      self$a$incorporate( self )
    },
    incorporate = function( a ){
      self$a <- a
    }
  )
)

tst <- test$new()
tst$contain()

tst$finalize()
