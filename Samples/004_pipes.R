library( cuRious )
library( microbenchmark )

# Transfer calls involve many checks to try to evade segmentation faults.
# These checks add up to a significant call overhead. For this reason, cuRious
# implements persistent reference objects called pipes. Pipes can be used and
# more importantly reused to do the same data transfer between tensors multiple
# times.

# Pipes take the same arguments as the transfer() function at initialization,
# and follow the same logic and limitations. In fact, the transfer() function
# is implemented as a thin wrapper around a temporary pipe:
print( transfer )

# A simple pipe transfer:
src <- tensor$new( matrix( 1:6, 2, 3 ) )
dst <- tensor$new( src, copy = FALSE )
pip <- pipe$new( src, dst )

pip$run()

print( dst$pull() )

# Pipes only check arguments when necessary and only once, significantly redu-
# cing call overhead compared to transfer() calls:
print( microbenchmark( transfer( src, dst ) ), times = 100 )
print( microbenchmark( pip$run() ), times = 100 )

# Pipes should be pre-defined when using tensors, and stored for re-use during
# runtime.

clean()
