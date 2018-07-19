library( cuRious )
library( microbenchmark )

# Transfer calls involve many checks to try to evade segmentation faults.
# These checks add up to a significant call overhead. For this reason, cuRious
# implements persistent reference objects called pipes. Pipes can be used, and
# more importantly reused to do the same data transfer between tensors multiple
# times.

# Pipes take the same arguments as the transfer() function at initialization,
# and follow the same logic, but have additional limitations. Pipes can not be
# used for multi-step transfers, usch as L0-L3, L3-L0 same device transfers, or
# L3-L3 between different devices.

# Pipes are realized as classes inheriting from the general fusion class.
# Fusions are operation involving one or multiple tensors as inputs and outputs.
# Every fusion requires a respective fusion context to function. Pipes are a
# a soft exception to this rule, as they can be created without a context. How-
# ever, the automatic creation of a pipe.context at every call incurs a
# significant overhead.

# Pipe context creation. Contexts are created with an inactive starting state
# by default, the same as any other context. To make them active at creation, a
# deployment level needs to be set. As we are going to use this context for L0-
# L0 transfers, level 1 suffices for now:
pip.cont <- pipe.context$new()$deploy( 1 )

# A simple pipe transfer:
src <- tensor$new( matrix( 1:6, 2, 3 ) )
dst <- tensor$new( src, copy = FALSE )
pip <- pipe$new( src, dst, context = pip.cont )

pip$run()

print( dst$pull() )

# Pipes only check arguments when necessary and only once, significantly redu-
# cing call overhead compared to transfer() calls:
print( microbenchmark( transfer( src, dst ) ), times = 100 )
print( microbenchmark( pip$run() ), times = 100 )

# Pipes should be pre-defined when using tensors, and stored for re-use during
# runtime.

clean()
