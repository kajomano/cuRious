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
# In fact dive() and surface are just shortcuts to calling transform with level
# 3 and 0 (default) respectively.
tens.x$dive
tens.x$surface

# Let's look at the individual levels:
# On level 0, the tensor is a common R object, as can be seen now:
tens.x$get.tensor

# The tensor's data should not be accessed with the get.tensor accessor.
# This is because different tensors and R objects sharing the same memopry space
# could be modified unintentionally, without R's lazy copy guarding them from
# this. Instead, the pull() or transfer() (more on this later) operation should
# be used whenever accessing the data in any way in the tensor. These operations
# create a hard copy of the data that is safe from unintentional modification.
# When creating a tensor, the initial information is also hard copied this way.

# An example of an unintentional modification:
not.really.copied.data <- tens.x$get.tensor
properly.copied.data   <- tens.x$pull()

print( not.really.copied.data )
print( properly.copied.data )

tens.x$push( rep(0, times = 10) )

print( not.really.copied.data )
print( properly.copied.data )

# On level 1, the information is stored as single precision floating point
# values. Conversion is needed because most GPUs have much stronger single
# precisions capabilities than double, and neural networks do not really require
# double precision anyway. This, however, introduces a processing step between
# the two levels. Calling transform() has no additional parameters on how this
# conversion should be handled. This is because transform() is not intended as a
# frequently called operation, its function is to allocate memory space for
# operations. However, transfer(), the function that moves data between tensors,
# does have a few extra parameters. Lets create another tensor, and move the
# data to level 1:
tens.y <- tensor$new( vect.x )
tens.y$transform( 1 )

# transfer() has many parameters, some of them active or not depending on the
# level of the source and target tensors/objects (sources and targets can also
# be plain R objects, which of course counts as level 0). When transferring data
# between level 1 or 2 and level 0, the conversion can be done on multiple
# threads. Let's see how the number of threads modify the runtime of the
# conversion:
microbenchmark( transfer( tens.x, tens.y, threads = 1L ), times = 10 )
microbenchmark( transfer( tens.x, tens.y, threads = 4L ), times = 10 )

# Running the conversion on multiple threads actually made things slower! This
# is because launching threads involves a certain overhead, which means that for
# small tensors, the overhead is bigger than the actual single threaded runtime.
# Lets try the same thing with bigger tensors (mind the uppercase X and Y):
big.n <- 10^6
tens.X <- tensor$new( rnorm( big.n ) )
tens.Y <- tensor$new( rnorm( big.n ) )
tens.Y$transform( 1 )

# TODO ====
# Itt vmi nem kerek

microbenchmark( transfer( tens.X, tens.Y, threads = 1L ), times = 10 )
microbenchmark( transfer( tens.X, tens.Y, threads = 4L ), times = 10 )

clean.global()
