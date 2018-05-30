library( cuRious )
library( microbenchmark )

obj1 <- matrix( as.numeric( 1:10^6 ), 1000, 1000 )
obj2 <- tensor$new( obj1, 1L, copy = FALSE )

subs1 <- as.integer( 1:obj2$dims[[2]] )
subs2 <- as.integer( 1:obj2$dims[[2]] )

fun <- function(){
  .Call( "cuR_transfer",
         obj1,
         obj2$ptr,
         0L,
         1L,
         "n",
         obj2$dims,
         obj2$dims,
         obj2$dims,
         subs1,
         subs2,
         # NULL,
         # NULL,
         NULL,
         NULL,
         NULL )
}

print( microbenchmark( fun() ), times = 1000 )
print( identical( obj1, obj2$pull() ) )

clean()
