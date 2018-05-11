# These functions are sanity checks and conversion functions for
# column subsetting in transfer() and algebraic (cuBLAS) calls

column.empty <- function( tens, cols ){
  list(
    off  = NULL,
    ptr  = NULL,
    dims = tens$dims
  )
}

column.range <- function( tens, cols ){
  if( any( !is.numeric( cols ),
           !length( cols ) == 2,
           as.logical( cols %% 1 ),
           cols[[2]] > tens$dims[[2]],
           cols[[2]] < cols[[1]],
           cols[[1]] < 0 ) ){
    stop( "Invalid column range subset" )
  }

  list(
    off  = as.integer( cols[[1]] ),
    ptr  = NULL,
    dims = c( tens$dims[[1]], as.integer( cols[[2]] - cols[[1]] + 1L ) )
  )
}

column.indiv <- function( tens, cols ){
  if( any( cols$type != "i",
           tens$level == 3,
           cols$level == 3 ) ){
    stop( "Invalid individual column subset" )
  }

  list(
    off  = NULL,
    ptr  = cols$ptr,
    dims = c( tens$dims[[1]], cols$l )
  )
}
