library( cuRious )
library( microbenchmark )

level.src <- 3L
level.dst <- 1L

mat <- matrix( as.numeric( 1:10^6 ), 1000, 1000 )
tens.src <- tensor$new( mat, level.src )
tens.dst <- tensor$new( mat, level.dst , copy = FALSE )

obj.src <- tens.src$ptr
obj.dst <- tens.dst$ptr

dims <- tens.src$dims

subs1 <- as.integer( 1:tens.src$dims[[2]] )
subs2 <- as.integer( 1:tens.src$dims[[2]] )

stream <- cuda.stream$new()
stream.ptr <- stream$ptr

fun <- function(){
  .Call( "cuR_transfer",
         obj.src,
         obj.dst,
         level.src,
         level.dst,
         "n",
         dims,
         dims,
         dims,
         subs1,
         subs2,
         # NULL,
         # NULL,
         NULL,
         NULL,
         NULL )
         # stream.ptr )
  # cuda.device.sync()
}

# fun()

print( microbenchmark( fun() ), times = 1000 )
print( identical( tens.src$pull(), tens.dst$pull() ) )

clean()
