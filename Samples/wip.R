library( cuRious )
library( microbenchmark )

stream  <- cuRious::stream$new( 3, 0 )
context <- cuRious::pipe.context$new( 4, stream )

src <- cuRious::tensor$new( 1.0, 3L )
# dst <- cuRious::tensor$new( src, 0L, copy = FALSE )
#
# pip <- cuRious::pipe$new( src, dst, context = context )
#
# pip$run()
#
# src$ptrs$tensor
# dst$ptrs$tensor

# clean()

# src.tensor,
# dst.tensor,
# src.level,
# dst.level,
# type,
# dims,
# src.dims        = NULL,
# dst.dims        = NULL,
# src.perm.tensor = NULL,
# dst.perm.tensor = NULL,
# src.span.off    = NULL,
# dst.span.off    = NULL,
# context.workers = NULL,
# stream.queue    = NULL
