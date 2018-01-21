# Although the main idea is to have most of the data residing in the GPU memory,
# all the training data can probably not be expected to fit there. When
# prototyping it is also sometimes easier to first implement something in pure
# R, which requires data transfer between the host and the device. For this
# case, cuRious tries to implement the data transfer as fast as possible. This
# script gives you a general feel how much time it can take to transfer data to
# and from the GPU.
library( cuRious )
library( microbenchmark )

# Create a vector
n <- 10^6
vect.x <- rnorm( n )

# Create a tensor, and move the information to the GPU
tens.x <- tensor$new( vect.x )
tens.x$dive()

# Define functions for a better microbenchmark print
memory.write <- function(){ tens.x$push( vect.x ) }
memory.read  <- function(){ tens.x$pull() }

# Check the speeds. Keep in mind, that if DEBUG_PRINTS are enabled, printing
# to the R console can take up time as well. However, it is not much compared
# to the overall transfer times. You can compile the package without the debug
# prints by commenting out the #define DEBUG_PRINTS 1 line in the debug.h file.
print( microbenchmark( memory.write(), times = 100 ) )
print( microbenchmark( memory.read(),  times = 100 ) )

# It might have caugth your attention that the read operation is almost twice as
# slow as the write. This is because memory needs to be allocated for a new
# R object when the function returns. Another aspect of this is that sometimes
# the memory allocation also triggers the R garbage collector, which takes
# forever to finish, causing orders of magnitude higher max times than means.
microbenchmark( gc(), times = 100 )

# To counteract both aspects, we can supply an R object of the correct dimensions
# which will serve as the output of the pull operation.
vect.y <- rep( 0, times = n )
memory.read  <- function(){ tens.x$pull( vect.y ) }

# Now the two operations should take up roughly the same amount of time
print( microbenchmark( memory.write(), times = 100 ) )
print( microbenchmark( memory.read(),  times = 100 ) )

# The placeholder vector also was overwritten
print( vect.y )

# Let's take things even further, and make tens.x staged. This creates a
# page-locked buffer for the tensor that can accelerate all push and pull calls
# for the cost of taking up memory in non-swappable address space. This is
# useful for tensors that will act as IO points between the CPU and GPU with
# regular push or pull operations. Use sparingly!
tens.x$create.stage()

# Let's check the speed with the staged tensor
print( microbenchmark( memory.write(), times = 100 ) )
print( microbenchmark( memory.read(),  times = 100 ) )

clean.global()
