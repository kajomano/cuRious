# This script shows how to move tensors to/from the GPU memory, and the loss of
# precision when doing so
library( cuRious )

# Create a vector
n <- 10
vect.x <- rnorm( n ) * 10^9

# Create a tensor, and move the information to the GPU
tens.x <- tensor$new( vect.x )
tens.x$dive()

# Recover (copy out) the vector
tens.x$pull()
# Should actually see some precision loss:
print( vect.x )
# This operation does not remove the data from GPU memory, as you can see:
tens.x$get.tensor

# Push new values into the stored vector on the GPU. This operation is preferred
# compared to reinitialization as it does not allocate new memory space, however
# you can only push a vector with the same length as the original
tens.x$push( rnorm( n ) )

# Let's copy the tensor. As you can see, it is a shallow copy, pointing to the
# same memory adress as its predecessor
tens.x2 <- tens.x
tens.x2$get.tensor

# Move the data back to the CPU memory. Let's also check what happens to the
# copy. The soft copying mechanism of R6 causes the copy to also surface. This
# is the intended and correct behavior. You can alos see that the $push() call
# did change the actual content
tens.x$surface()
tens.x2$get.tensor

clean.global()
