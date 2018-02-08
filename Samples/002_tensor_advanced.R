# The previous script illustrated a very basic usage of tensors. If only life
# was that simple! This script shows the different levels (positions in memory)
# the tensors can inhibit, and how to move data between tensors on these levels.
library( cuRious )
library( microbenchmark )

# Create a vector and a tensor
vect.x <- rnorm( 10 )
tens.x <- tensor$new( vect.x )

# Tensors can actually be in 4 different stages (levels), which are defined by
# the combination of how the data is stored, and where. The 4 levels are the
# following:
# - Level 0: The tensor is an R object, a matrix or a vector. It is stored in
# host memory as double precision floating point values.
# - Level 1: The tensor is still in host memory, but it is converted to single
# precision floating point values.
# - Level 2: The tensor is in the host memory, but inhibits a page-locked space,
# and is stored as single precision floats.
# - Level 3: The tensor is stored in the device memory as single precision
# floats.

# When calling dive() or surface(), the tensor goes through all of these levels.
# The tensor can be individually set on any level by calling transform( level ).
# In fact dive() and surface are just wrappers to calling transform() with level
# 3 and 0 (default) respectively.
tens.x$dive
tens.x$surface

# Let's look at the individual levels:
# On level 0, the tensor is a common R object, as can be seen now:
tens.x$get.obj

# The tensor's data should not be accessed with the get.obj accessor.
# This is because different tensors and R objects sharing the same memopry space
# could be modified unintentionally, without R's lazy copy guarding them from
# this. Instead, the pull() or transfer() (more on this later) operation should
# be used whenever accessing the data in any way in the tensor. These operations
# create a hard copy of the data that is safe from unintentional modification.
# When creating a tensor, the initial information is also hard copied this way.

# An example of an unintentional modification:
not.really.copied.data <- tens.x$get.obj
properly.copied.data   <- tens.x$pull()

print( not.really.copied.data )
print( properly.copied.data )

tens.x$push( rep(0, times = 10) )

print( not.really.copied.data )
print( properly.copied.data )

# On level 1, the information is stored as single precision floating point
# values. Conversion is needed because most GPUs have much stronger single
# precisions capabilities than double, and neural networks do not really require
# double precision. This, however, introduces a processing step between
# the two levels. Lets see how processing influences transfer times on bigger
# tensors. You can relocate a tensor to level 1 by calling transform( level=1 ),
# and move the information between tensors by calling transfer( source, dest ).
big.n <- 10^6
tens.X <- tensor$new( rnorm( big.n ) )
tens.Y <- tensor$new( rnorm( big.n ), 1 )

microbenchmark( transfer( tens.X, tens.Y ), times = 10 ) # L0 -> L0 transfer
microbenchmark( transfer( tens.X, tens.X ), times = 10 ) # L0 -> L1 transfer

# On more modern machines the transfer that also includes the processing is
# actually faster than the one that doesn't. This is because the L0->L1 transfer
# reads the same amount of information as it's peer, but only writes half as
# much, since floats are stored on half the space. The overall runtime is set
# on modern processors not by the time spent on converting from double to float,
# but by waiting on memory reads and writes. Even though the preprocessing
# does not slow down the memory transfer however, it is still sometimes useful
# to pre-convert data to float to be able to skip this step in later operations.
# Such use is for example when handling training data, as it only needs to be
# converted once, but will be accessed many times.

# Level 2 is much the same as level 1, but the allocated memory is page-locked.
# This means that it can not be removed from the memory to make space for more
# data by swapping it to the disk. Page locked memory is a very inflexible
# resource allocation, and should be used sparingly. It does, however, allow us
# to transfer data to level 3 asychronously. If transferring data to or from
# level 3, and the other side is page-locked (level 2), the transfer will be
# asynchronous with regards to the host if an active stream is also supplied.
# Here is an example:

stream <- cuda.stream$new()
stream$activate()

tens.X$transform( 2 )
tens.Y$transform( 3 )

# sync
print( microbenchmark( transfer( tens.X, tens.Y ), times = 10 ) )
# async
print( microbenchmark( transfer( tens.X, tens.Y, stream=stream ), times = 10 ) )

# The asynchronous call is much faster, because it returns before the actual
# data transfer is completed. This can be useful for overlapping data transfers
# with computations, also known as data streaming. This will be shown in a later
# sample script.

clean.global()
