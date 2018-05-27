library( cuRious )
library( microbenchmark )

l <- as.integer( 10 )
src <- as.numeric( 1:l )
dst <- as.numeric( 2:(l + 1 ) )

tracemem( src )

fun <- function(){
  .Call( "cuR_transfer",
         src,
         dst,
         0L,
         0L,
         "n",
         c( 1L, l ),
         c( 1L, l ),
         c( 1L, l ),
         NULL,
         NULL,
         NULL,
         NULL,
         NULL )
}

fun()

microbenchmark( fun() )

dst[[1]]

# SEXP src_ptr_r,
# SEXP dst_ptr_r,
# SEXP src_level_r,
# SEXP dst_level_r,
# SEXP type_r,
# SEXP src_dims_r,
# SEXP dst_dims_r,
# SEXP dims_r,
# SEXP src_perm_ptr_r,  // Optional
# SEXP dst_perm_ptr_r,  // Optional
# SEXP src_span_off_r,  // Optional
# SEXP dst_span_off_r,  // Optional
# SEXP stream_ptr_r
