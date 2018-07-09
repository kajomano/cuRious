thr <- 10^-6

test.thr.equality <- function( obj1, obj2 ){
  all(
    obj1 >= obj2 * ( 1 - thr ),
    obj2 >= obj1 * ( 1 - thr ),
    obj1 <= obj2 * ( 1 + thr ),
    obj2 <= obj1 * ( 1 + thr )
  )
}
