# This script shows how matrices are stored in memory, matrix subsetting, and
# the different types of tensors.
library( cuRious )

# Large matrices often need to be split into smaller chunks. R's matrix
# subsetting is slow because it allocates new space for the subset, and copies
# the data. Most of the time we are transferring information anyway, so the
# subsetting step can be incorporated into the transfer() function. To this
# extent, all transfers calls are subsettable to some degree.

# There is a catch however: the way R and CUDA expects data is in
# column-major format, in a continous memory space. Here is an example how this
# looks:
#
# R matrix:     | 1 3 5 |
#               | 2 4 6 |
#
# Memory image: | 1 2 3 4 5 6 |
#
# Subsetting rows from this data would require read operations from memroy add-
# resses that are nrow() places apart. This can not be vectorized effectively,
# and so is very slow compared to column subsetting. For this reason, cuRious
# only implements column-wise subsetting. For splitting large datasets, it is
# recommended to store the data in matrices where the rows are the dimensions,
# and the columns are the individual observations.

# Curious implements two forms of column subsetting: range-based subsets and
# individually selected columns.

# Range-based subsetting ====
# Subsetting a continous range of columns can be done on any transfer calls, by
# supplying a list of two integers as subset, corresponding to the first and
# last column ids respectively. Both source and destination can be subsetted
# this way:

tens.X <- tensor$new( matrix( as.numeric( 1:6 ), 2, 3 ), 1L )
tens.Y <- tensor$new( matrix( as.numeric( 0   ), 2, 3 ), 1L )

transfer( tens.X, tens.Y, src.cols = c( 1L,2L ), dst.cols = c( 2L,3L ) )

print( tens.Y$pull() )

# Individual subsetting ====
# Individual columns can be subsetted an reordered by supplying an integer
# tensor as subset. Both source and destination can be subsetted, but this form
# of subsetting only works on host-host transfers (transfers not involving L3
# tensors):

tens.src.sub <- tensor$new( c(1L,3L,2L) )
tens.dst.sub <- tensor$new( c(3L,2L,1L) )

tens.Y$push( matrix( as.numeric( 0 ), 2, 3 ) )
transfer( tens.X, tens.Y, src.cols = tens.src.sub, dst.cols = tens.dst.sub )
print( tens.Y$pull() )

# Tensor types ====
# So far we only used tensors that contained numeric data. However, cuRious
# actually supports 3 kind of tensors (named after the R storage types):
#
# Numeric: R side stored as double, C side stored as float
# Integer: Both sides stored as integers
# Logical: R side stored as integer, C side stored as bool
#
# The type of the tensor is set at initialization, by supplying the constructor
# with the correct type of object, and can not be changed later. Transfers are
# not possible between different types of tensors or objects. Tensor types are
# abbreviated to the first letter of the R type, so "n", "i" and "l"
# respectively:

tens.num <- tensor$new( rep( 1, times = 6 ) )
tens.int <- tensor$new( rep( 1L, times = 6 ) )
tens.log <- tensor$new( rep( TRUE, times = 6 ) )

print( tens.num$type )
print( tens.int$type )
print( tens.log$type )

clean()
