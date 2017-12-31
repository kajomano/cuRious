# Although the main idea is to have all the data residing in the GPU memory, and
# not to have any CPU memory transfer during training, sometimes for rapid
# prototyping it is much easier to first implement something in pure R. For this
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
# to the overall transfer times.
microbenchmark( memory.write(), times = 100 )
microbenchmark( memory.read(),  times = 100 )

clean.global()
