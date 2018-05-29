library( cuRious )
library( microbenchmark )

# Tensors are objects in host (CPU) or device (GPU) memory containing data in
# the form of numeric, integer or boolean vectors or matrices.

# Let's create a simple tensor:
tens <- tensor$new(
  # The data argument can contain the original data with which the tensor will
  # be initialized. Data can be either a simple R matrix or vector (object from
  # now on), or another tensor. If unset, the tensor will be initialized with
  # 0-s or FALSE-es.
  data   = NULL, # Object or tensor

  # The level argument can be set to define the initial level the tensor will
  # inhibit. This argument can be defined even when the data argument is set.
  # If left unset, the default is L0 for objects or undefined data, or the
  # same level as the data tensor. Levels govern how the tensor is stored, and
  # will be explained in detail in the next sample.
  level  = 0L, # Integer in the range 0:3

  # Dimensions can be defined by setting the dims argument with an integer
  # vector containing the number of c( rows, columns ) of the desired tensor.
  # Vectors are row vectors in cuRious (c( 1L, ncol )). If data is set, the
  # dimensions will be copied from it. Undefined data and dims will produce an
  # error. If data is set, setting this argument will also produce an error.
  dims   = c( 1L, 1L ), # Vector of 2 integers > 0

  # Type is the R type of the stored data. It can take the values "numeric",
  # "integer" and "logical", or any form of shorthand of these words. If data is
  # set, the type will be copied from it. Undefined data and type will produce
  # an error. If data is set, setting this argument will also produce an error.
  type   = c( "num", "int", "log" ),

  # The copy flag decides wether actual data will be used for initialization. If
  # set to FALSE, the tensor will be initialized with 0-s or FALSE-es.
  copy = TRUE,

  # Device is important in multi-gpu environments. Device can be set for any
  # level, but only becomes important on L3. If unset, the default device will
  # be set. The use of multiple devices will be shown in a later sample script.
  device = NULL # Integer in the range 0:( devices-1 )
)

# The contents of a tensor can be accessed by the $pull() function, and over-
# written by the $push() function of the tensor. These functions are intended
# only for debugging and experimentational purposes. We will see how to modify
# the tensor contents in implementations in a later sample script. When using
# $push(), the type and dimensions of the supplied object or tensor must match
# the type and dimensions of the tensor, otherwise an error will be produced.

# Let's check the contents:
print( tens$pull() )

# And overwrite them:
tens$push( 1 )

# Tensors are intended to be reference objects. By copying a tensor, only
# the reference is copied, but not the actual data within. This means that any
# modifications to the original tensor will be also visible in the copy, and
# vice-versa.

# Tensor creation from an R object:
tens.wrap <- tensor$new( 1:2 )

# Copying only the handle:
tens.copy <- tens.wrap

# Properly duplicating the data:
tens.dupl <- tensor$new( tens.wrap )

# Let's change the content of the tensor:
tens.copy$push( 3:4 )

# As can be seen, the copy also changed, while the properly duplicated tensor
# did not:
print( tens.copy$pull() )
print( tens.dupl$pull() )

# Tensor contents can also be accessed directly by calling the $obj active
# binding, if the tensor is on L0:
tens$obj <- 0
print( tens$obj )

# Tensors should be considered as placeholders, which functions can read from
# and write to. A good program utilizing tensors should minimize the creation
# (and destruction) of tensors during runtime, as memory allocation and
# deallocation can incur significant overhead. Instead, try to preallocate
# most of what will be needed, and try to reuse as much as possible.

# Tensor creation overhead is considerable even on a very small tensor. Larger
# tensors require even more time:
print( microbenchmark( tensor$new( 1:100 ), times = 100 ) )

clean()
