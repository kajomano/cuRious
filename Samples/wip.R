library( cuRious )
library( microbenchmark )

stream  <- cuRious::stream$new( 3, 0 )
context <- cuRious::pipe.context$new( 4, stream )

# src <- cuRious::tensor$new( rep( 1.0, times = 10^6 ) )
src <- cuRious::tensor$new( 1.0 )
dst <- cuRious::tensor$new( src, copy = FALSE )

fun <- function(){
  cuRious::transfer.ptr(
    src$ptrs$tensor,
    dst$ptrs$tensor,
    src$level,
    dst$level,
    src$type,
    src$dims,
    context.workers = context$ptrs$workers#,

    # stream.queue    = stream$ptrs$queue
  )
}

fun()

src$ptrs$tensor
dst$ptrs$tensor

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
