library( cuRious )

# Tensors can inhibit 4 levels: 0, 1, 2 and 3. These levels define whether the
# tensor data is stored on the host, or the device, and in what format. The 4
# levels are the following:
#
#                        ############
#                        #  R-side  #
#                        ############
#
# L0: The tensor$ptr is an R object, a matrix or a vector.
#
#        type: | numeric | integer | logical |
#              |-----------------------------|
#     storage: | double  |   int   |   int   |
#
# ---------------------------------------------------------
#
# L1: The tensor$ptr is a pointer pointing to a memory
#     address in host memory.
#
#        type: | numeric | integer | logical |
#              |-----------------------------|
#     storage: |  float  |   int   |  bool   |
#
# ---------------------------------------------------------
#
# L2: The tensor$ptr is a pointer pointing to a memory
#     address in host memory. The memory is pinned, and
#     directly accessible by any device.
#
#        type: | numeric | integer | logical |
#              |-----------------------------|
#     storage: |  float  |   int   |  bool   |
#
# =========================================================
# =================== Host-device border ==================
# =========================================================
#
# L3: The tensor$ptr is a pointer pointing to a memory
#     address in device memory.
#
#        type: | numeric | integer | logical |
#              |-----------------------------|
#     storage: |  float  |   int   |  bool   |
#
#                      ###############
#                      #  CUDA-side  #
#                      ###############

# When transferring data from an R object to a device, the data needs to through
# all of these levels. Data transfers will be explained in more detail in the
# next sample script.

# The level of a tensor can be set at initialization, or changed through the
# $level active binding. Let's move some data to the device:
tens <- tensor$new( rnorm( 10 ), level = 3L )

# The tensor ptr is an actual pointer pointing to device memory:
print( tens$ptrs$tensor )

# Direct assignement is not allowed on any level other than L0:
try( tens$obj <- 1 )

# GPU computations usually involve numeric matrices. As not many GPU-s have
# strong double precision floating-point capabilities, numeric data is converted
# from double-precision to single-precision between L0-L1 data transfers. This
# can result in a loss of precision.

# Some values are not the same:
R.vect <- rnorm( 10 ) * 10^9
tens$push( R.vect )
print( R.vect )
print( tens$pull() )

# The device memory allocation can be freed by moving the tensor back to host
# memory (L0, L1, L2), or all memory can be freed thorugh the $destroy()
# operation. Destroying a tensor does not actually remove the reference object,
# but makes any further interaction produce an error:
tens$destroy()
try( tens$obj )

clean()
