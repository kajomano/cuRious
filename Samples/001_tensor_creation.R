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
  data   = 0, # Object or tensor

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

  # The initialization method governs how to process the data argument if it is
  # supplied. The default is to copy the information. "mimic" copies the meta-
  # information such as level, dimensions, type and device, but not the actual
  # data. The tensor will be initilaized with 0-s or FALSE-es in this case.
  # "wrap" can be only used if data is an object. Wrapping an object does not
  # duplicate the data from the object, but wraps it into a tensor. We will
  # see the benefits of this use later.
  init   = c( "copy", "mimic", "wrap" ),

  # Device is important in multi-gpu environments. Device can be set for any
  # level, but only becomes important on L3. If unset, the default device will
  # be set. The use of multiple device will be shown in a later sample script.
  device = NULL # Integer in the range 0:( devices-1 )
)

# The contents of a tensor can be accessed by the $pull() function, and over-
# written by the $push() function of the tensor. These functions are intended
# only for debugging and experimentational purposes. We will see how to modify
# the tensor contents in implementations in a later sample script. When using
# $push(), the type and dimensions of the supplied objest must match the type
# and dimensions of the tensor, otherwise an error will be produced.

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

# As can be seen, the copy also changed, while the properly duplicated data
# did not:
print( tens.copy$pull() )
print( tens.dupl$pull() )

# Tensor contents can also be accessed directly by calling the $ptr active
# binding, if the tensor is on L0:
print( tens$ptr )

# Assignement is also possible this way, however, the tensor will gain the
# read.only flag, preventing any modifications to it. Since this assignement
# does not duplicate data, the read-only lock is to ensure that no objects are
# modified silently as a byproduct of a direct access. The $push() and $pull()
# operations are safe from these side-effects, and properly duplicate data.

# These assignements will produce an error:
tens$push( 2 ) # Safe duplicated assignement
tens$ptr <- 3  # Direct assignement locks the tensor to read-only
try( tens$push( 4 ) ) # Error

# Direct assignements however work even after read-only locks:
tens$ptr <- 4  # No error

# Initializing a tensor by wrapping an object also produces read-only tensors:
tens.wrap <- tensor$new( 1, init = "wrap" )
try( tens.wrap$push( 2 ) ) # Error

# To prevent unintended modification, NEVER store tensor contents from direct
# access:
tens.never       <- tensor$new( 0 )
tens.never.data  <- tens.never$ptr
tens.never$push( 1 )
print( tens.never.data ) # The stored data also changes

# Tensors should be considered as placeholders, which functions can read from
# and write to. A good program utilizing tensors should minimize the creation
# (and destruction) of tensors during runtime, as memory allocation and
# deallocation can incur significant overhead. Instead, try to preallocate
# most of what will be needed, and try to reuse as much as possible.

# Tensor creation overhead on a very small tensor. Larger tensors require
# even more time:
microbenchmark( tensor$new( 1:100 ), times = 100 )

clean()
