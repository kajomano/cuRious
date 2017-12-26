# This script shows how to move vectors to/from the GPU memory
library( cuRious )

# Create a vector
n <- 10
test.vect <- rnorm( n ) * 10^6

# "Dive" the object to the GPU memory. Keep in mind that this results in a
# conversion from double to float, thereby losing precision
test.vect.pointer <- dive( test.vect )

# Recover the vector
surface( test.vect.pointer )

# Should actually see some precision loss
print( test.vect )
