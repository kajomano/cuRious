# This script shows how to move tensors to/from the GPU memory, and the loss of
# precision when doing so.
library( cuRious )

# Create a vector
vect.x <- rnorm( 10 ) * 10^9

# In most literature the CPU is referred to as the host, while the GPU
# as device. Host memory is system memory on the motherboard, while device
# memory is integrated memory on the graphics card. Let's create a tensor from
# the vector, and relocate (dive) the information to the device. Tensors are R6
# objects that hold the pointer to the data wherever it may be located.
tens.x <- tensor$new( vect.x )
tens.x$dive()

# Pull (copy out) the vector back to R. You should actually see some precision
# loss, the cause of this will be explained in the next script.
print( tens.x$pull() )
print( vect.x )

# The pull operation does not remove the data from GPU memory, as you can see:
tens.x$get.tensor

# Push (copy in) new values into the stored vector on the device. This operation
# is preferred compared to reinitialization as it does not allocate new memory.
# You can only push a vector with the same length as the original (matrix with
# the same dimensions).
tens.x$push( rep( 0, times = 10 ) )

# Let's copy the tensor object. As you can see, it is a shallow copy, pointing
# to the same memory adress as its predecessor:
tens.y <- tens.x
tens.y$get.tensor

# Move back (surface) the data to the host memory. Let's also check what happens
# to the copy. The soft copying mechanism of R6 causes the copy to also surface
# (this is intended). You can also see that the previous push() call did change
# the actual content.
tens.x$surface()
tens.y$pull()

clean.global()
