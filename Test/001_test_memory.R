# This script shows how to move tensors to/from the GPU memory, and the loss of
# precision when doing so
library( cuRious )

# Create a vector
n <- 10
vect.x <- rnorm( n ) * 10^9

# Create a vect object, and move the information to the GPU
vect.x.obj <- tensor$new( vect.x )
vect.x.obj$dive()

# Recover (copy out) the vector
vect.x.obj$pull()
# Should actually see some precision loss:
print( vect.x )
# This operation does not move the data out of the GPU memory, as you can see:
vect.x.obj$get.tensor

# Push new values into the stored vector on the GPU. This operation is preferred
# compared to reinitialization as it does not allocate new memory space, however
# you can only push a vector with the same length as the original
vect.x.obj$push( rnorm( n ) )

# Move the data back to the CPU memor. Watch for the message from finalizing
# (cleaning up) the GPU memory
vect.x.obj$surface()
gc()

# Check if the values actually changed with $push()
vect.x.obj$get.tensor
