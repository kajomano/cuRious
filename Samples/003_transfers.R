library( cuRious )

# Information between tensors can be transferred with the appropriately called
# transfer() function. Both the source and destination tensors can inhibit any
# level, however, some restrictions to additional functionality applies, as we
# will see shortly.

# A simple transfer invocation:
src <- tensor$new( matrix( 1:6, 2, 3 ) )
dst <- tensor$new( src, init = "mimic" )

transfer( src, dst )

print( dst$pull() )

# Transfer() is also used when invoking the $pull() or $push() operations.

# Large matrices often need to be split into smaller chunks. To this extent, all
# transfer() calls are subsettable to some degree. There is a catch however: the
# way R and CUDA expects data is in column-major format, in a continous memory
# space. Here is an example how this looks:
#
# R matrix:     | 1 3 5 |
#               | 2 4 6 |
#
# Memory image: | 1 2 3 4 5 6 |
#
# Subsetting rows from this data would require read operations from memory add-
# resses that are nrow places apart. As nrow is not known a-priori to compi-
# lation, such subsetting can not be vectorized effectively, and is very slow
# compared to column subsetting. For this reason, cuRious only implements
# column-wise subsetting. For splitting large datasets, it is recommended to
# store the data in matrices where the rows are the dimensions, and the columns
# are the individual observations.
#
# Vectors do not suffer from such problems, as the memory-image would be
# continous for both row- and column-vectors. In fact, the memory image would
# not change when transposing a vector. However, to stay in line with the logic
# of column-only subsetting but allow for vector subsetting, vectors are consi-
# dered as row vectors.

# Curious implements two forms of column subsetting: range-based subsets and
# individually selected columns.

# Range-based subsetting (span) ====
# Subsetting a continous range of columns can be done on any transfer call, by
# supplying a list of two integers as subset, corresponding to the first and
# last column ids respectively. Both source and destination can be subsetted
# this way:

dst$clear()

transfer( src, dst, src.span = c( 1L, 2L ), dst.span = c( 2L, 3L ) )

print( src$pull() )
print( dst$pull() )

# Individual subsetting (permutaion) ====
# Individual columns can be subsetted and reordered by supplying an integer
# tensor as subset. Both source and destination can be subsetted, but this form
# of subsetting only works on the following transfers:
#
#      | L0 | L1 | L2 | L3 |
#   ---|----|----|----|----|
#   L0 | *  | *  | *  |    |
#   ---|----|----|----|----|
#   L1 | *  | *  | *  |    |
#   ---|----|----|----|----|
#   L2 | *  | *  | *  |    |
#   ---|----|----|----|----|
#   L3 |    |    |    | *  |
#
# In the case of L3-L3 permutated subsets, the permutation tensors also need to
# be on L3. In any other case, the permutation tensors can be L0, L1 and L2:

src.perm <- tensor$new( c(1L, 3L, 2L) )
dst.perm <- tensor$new( c(3L, 2L, 1L) )

dst$clear()

transfer( src, dst, src.perm = src.perm, dst.perm = dst.perm )

print( src$pull() )
print( dst$pull() )

# Permutated subsets can also be used with spans:
dst$clear()

transfer( src,
          dst,
          src.perm = src.perm,
          dst.perm = dst.perm,
          src.span = c( 1L, 2L ),
          dst.span = c( 2L, 3L ) )

print( src$pull() )
print( dst$pull() )

clean()
