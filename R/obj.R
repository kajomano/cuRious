# Dimension encoder
obj.dims <- function( obj ){
  obj <- check.obj( obj )

  switch(
    class( obj )[[1]],
    matrix  = c( nrow( obj ), ncol( obj ) ),
    numeric = c( 1L, length( obj ) ),
    integer = c( 1L, length( obj ) ),
    logical = c( 1L, length( obj ) )
  )
}

# Get storage type
obj.type <- function( obj ){
  obj <- check.obj( obj )

  switch(
    class( obj )[[1]],
    matrix     = {
      mode <- storage.mode( obj )
      if( mode == "double" ) mode <- "numeric"

      if( !(mode %in% types) ){
        stop("Invalid type")
      }

      names(types)[[ which( types == mode ) ]]
    },
    numeric = "n",
    integer = "i",
    logical = "l"
  )
}

# Create a placeholder object
obj.create <- function( dims, type = "n" ){
  dims <- check.dims( dims )
  type <- check.type( type )

  if( dims[[1]] == 1 ){
    vector( types[[type]], dims[[2]] )
  }else{
    matrix( vector( types[[type]], 1 ),
            nrow = dims[1],
            ncol = dims[2] )
  }
}
