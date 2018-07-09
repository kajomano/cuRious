library( cuRious )
library( microbenchmark )

thr <- 10^-6

test.thr.equality <- function( obj1, obj2 ){

  switch(
    obj.type( obj1 ),
    numeric = {
      all(
        obj1 >= obj2 * ( 1 - thr ),
        obj2 >= obj1 * ( 1 - thr ),
        obj1 <= obj2 * ( 1 + thr ),
        obj2 <= obj1 * ( 1 + thr )
      )
    },
    identical( obj1, obj2 )
  )
}
